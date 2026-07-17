"""Option #3 SWE solver: async in-flight stepping with coarse host synchronization."""
# pyright: reportMissingImports=false,reportPrivateUsage=false,reportUnknownArgumentType=false,reportUnknownMemberType=false,reportUnknownVariableType=false,reportUnusedExpression=false

from __future__ import annotations

import math
import sys
import time as _time
from typing import TYPE_CHECKING

import kp
import numpy as np

from .swe_gpu import SWESolver, _pc_cfl_accum, _pc_cfl_reduce

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


# Thresholds for CFL time step and simulation progression
_CFL_MIN_DT = 1e-10  # Minimum CFL dt before raising error
_TIME_EPSILON = 1e-12  # Epsilon for time/dt comparisons to avoid division by zero


class SWESolverAsyncSyncWindow(SWESolver):
    """Async variant that submits many timesteps and synchronizes at output boundaries."""

    def _build_algorithms(self, spv: dict[str, bytes]) -> None:
        N, E, WG = self.N, self.E, self._work_group_size
        wg_e = (int(np.ceil(E / WG)), 1, 1)
        wg_c = (int(np.ceil(N / WG)), 1, 1)
        g, dt = self._g, self._dry_tol

        self._algo_flux_dtbuf = self._mgr.algorithm(
            self._all_tensors,
            spv["flux_dtbuf"],
            workgroup=wg_e,
            spec_consts=[],
            push_consts=[float(E), 0.0, g, dt, self._cfl, 0.0],
        )
        self._algo_update_dtbuf = self._mgr.algorithm(
            self._all_tensors,
            spv["update_dtbuf"],
            workgroup=wg_c,
            spec_consts=[],
            push_consts=[float(N), 0.0, g, dt, self._cfl, 0.0],
        )
        self._algo_cfl_accum = self._mgr.algorithm(
            self._all_tensors,
            spv["cfl_accum"],
            workgroup=wg_e,
            spec_consts=[],
            push_consts=_pc_cfl_accum(E, g, dt, self._cfl),
        )
        self._algo_cfl_reduce = self._mgr.algorithm(
            self._all_tensors,
            spv["cfl_reduce"],
            workgroup=wg_c,
            spec_consts=[],
            push_consts=_pc_cfl_reduce(N, g, dt, self._cfl),
        )
        self._algo_cfl_resolve = self._mgr.algorithm(
            self._all_tensors,
            spv["cfl_resolve"],
            workgroup=(1, 1, 1),
            spec_consts=[],
            push_consts=[0.0, 0.0, 0.8, 0.0, 0.0],
        )

        self._algo_source = None
        if "source_dtbuf" in spv:
            self._source_dtbuf_spv = spv["source_dtbuf"]

    def _build_source_algo(self, src_dh_per_sec: NDArray[np.float32]) -> None:
        """Build source kernel that reads dt from dt buffer."""
        N, WG = self.N, self._work_group_size
        wg_c = (int(np.ceil(N / WG)), 1, 1)
        self.t_source_dtbuf = self._mgr.tensor(self._as_f32_1d(src_dh_per_sec))
        self._source_tensors = [*self._all_tensors, self.t_source_dtbuf]
        self._mgr.sequence().record(kp.OpTensorSyncDevice([self.t_source_dtbuf])).eval()
        self._algo_source = self._mgr.algorithm(
            self._source_tensors,
            self._source_dtbuf_spv,
            workgroup=wg_c,
            spec_consts=[],
            push_consts=[float(N), 0.0],
        )

    def run(
        self,
        t_end: float,
        output_interval_s: float,
        dt_max: float,
        dt_init: float,
        cfl_interval: int,
        progress: bool = True,
        source_rate: NDArray[np.float32] | None = None,
        source_fn: Callable[[float], NDArray[np.float32]] | None = None,
        resume: bool = False,
        t_start: float = 0.0,
    ) -> tuple[list[NDArray[np.float32]], list[float]]:
        """Run simulation with async sequences and coarse synchronization cadence."""
        if source_fn is not None:
            return super().run(
                t_end=t_end,
                output_interval_s=output_interval_s,
                dt_max=dt_max,
                dt_init=dt_init,
                cfl_interval=cfl_interval,
                progress=progress,
                source_rate=source_rate,
                source_fn=source_fn,
                resume=resume,
                t_start=t_start,
            )

        N, E = self.N, self.E
        g, dry_tol = self._g, self._dry_tol
        cfl = self._cfl
        cfl_safety = 0.8

        if not resume:
            self.reset(self._h0, np.zeros(N, np.float32), np.zeros(N, np.float32))

        use_gpu_source = False
        src_dh_per_sec: NDArray[np.float32] | None = None
        if source_rate is not None:
            src_dh_per_sec = np.asarray(source_rate, dtype=np.float32) * (1.0 / self._area).astype(
                np.float32
            )
            if self._algo_source is None and hasattr(self, "_source_dtbuf_spv"):
                self._build_source_algo(src_dh_per_sec)
            elif self._algo_source is not None and hasattr(self, "t_source_dtbuf"):
                self._upload(self.t_source_dtbuf, src_dh_per_sec)
            if self._algo_source is not None:
                use_gpu_source = True

        t_offset = t_start if resume else 0.0
        if resume:
            snapshots: list[NDArray[np.float32]] = [self.download_h().copy()]
            snap_times: list[float] = [t_start]
        else:
            snapshots = [self._h0.copy()]
            snap_times = [0.0]

        t_sim = t_offset
        step = 0
        dt = dt_init
        next_output_time = t_offset + output_interval_s
        wall_start = _time.monotonic()
        last_log_wall = wall_start
        log_interval = 10.0

        while t_sim < t_end:
            dt_cap = min(t_end - t_sim, dt_max)
            dt = min(dt, dt_cap)

            if step % cfl_interval == 0:
                np.asarray(self.t_dtbuf.data())[0] = self._SENTINEL_F
                cfl_seq = self._mgr.sequence()
                cfl_seq.record(kp.OpTensorSyncDevice([self.t_dtbuf]))
                cfl_seq.record(
                    kp.OpAlgoDispatch(self._algo_cfl_accum, _pc_cfl_accum(E, g, dry_tol, cfl))
                )
                cfl_seq.record(
                    kp.OpAlgoDispatch(self._algo_cfl_reduce, _pc_cfl_reduce(N, g, dry_tol, cfl))
                )
                cfl_seq.record(
                    kp.OpAlgoDispatch(
                        self._algo_cfl_resolve,
                        [float(dt), float(dt_cap), float(cfl_safety), 0.0, 0.0],
                    )
                )
                cfl_seq.record(kp.OpTensorSyncLocal([self.t_dtbuf]))
                cfl_seq.eval()
                dt = float(np.array(self.t_dtbuf.data(), dtype=np.float32)[0])

                if dt < _CFL_MIN_DT:
                    raise RuntimeError(f"CFL dt too small at step {step}: {dt:.3e}")
            else:
                np.asarray(self.t_dtbuf.data())[0] = np.float32(dt)
                self._mgr.sequence().record(kp.OpTensorSyncDevice([self.t_dtbuf])).eval()

            dt = min(dt, t_end - t_sim)

            step_seq = self._mgr.sequence()
            step_seq.record(
                kp.OpAlgoDispatch(self._algo_flux_dtbuf, [float(E), 0.0, g, dry_tol, cfl, 0.0])
            )
            step_seq.record(
                kp.OpAlgoDispatch(
                    self._algo_update_dtbuf,
                    [float(N), 0.0, g, dry_tol, cfl, 0.0],
                )
            )
            step_seq.record(
                kp.OpAlgoDispatch(self._algo_flux_dtbuf, [float(E), 0.0, g, dry_tol, cfl, 1.0])
            )
            step_seq.record(
                kp.OpAlgoDispatch(
                    self._algo_update_dtbuf,
                    [float(N), 0.0, g, dry_tol, cfl, 1.0],
                )
            )
            if use_gpu_source:
                step_seq.record(kp.OpAlgoDispatch(self._require_algo_source(), [float(N), 0.0]))
            step_seq.eval()

            t_sim += dt
            step += 1

            now = _time.monotonic()
            if progress and now - last_log_wall >= log_interval:
                wall_elapsed = now - wall_start
                pct = t_sim / t_end * 100.0 if t_end > 0 else 0.0
                eta = (
                    (wall_elapsed / t_sim) * max(t_end - t_sim, 0.0)
                    if t_sim > _TIME_EPSILON
                    else float("inf")
                )
                eta_str = f"{eta:.1f}s" if math.isfinite(eta) else "inf"
                sys.stdout.write(
                    f"[solver][step {step}] sim={t_sim:.2f}/{t_end:.2f}s "
                    f"({pct:.1f}%) dt={dt:.4f} eta~{eta_str}\n"
                )
                sys.stdout.flush()
                last_log_wall = now

            if t_sim >= next_output_time:
                h_snap = self.download_h()
                if not np.isfinite(h_snap).all():
                    snapshots.append(np.nan_to_num(h_snap, nan=0.0, posinf=0.0).copy())
                    snap_times.append(t_sim)
                    break
                if progress:
                    wall_elapsed = _time.monotonic() - wall_start
                    snap_idx = len(snapshots)
                    sys.stdout.write(
                        f"[solver] snapshot {snap_idx} captured at t={t_sim:.2f}s "
                        f"(wall={wall_elapsed:.1f}s)\n"
                    )
                    sys.stdout.flush()
                snapshots.append(h_snap.copy())
                snap_times.append(t_sim)
                next_output_time += output_interval_s

        h_final = self.download_h()
        if not snapshots or not np.allclose(snapshots[-1], h_final):
            snapshots.append(h_final.copy())
            snap_times.append(t_sim)

        return snapshots, snap_times


__all__ = ["SWESolverAsyncSyncWindow"]
