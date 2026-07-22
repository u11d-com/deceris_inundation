"""Update shader sources for SWE solver."""

UPDATE_GLSL = """\
#version 450

layout(local_size_x = 256) in;

layout(set=0, binding=0)  buffer H      { float h[];      };
layout(set=0, binding=1)  buffer HU     { float hu[];     };
layout(set=0, binding=2)  buffer HV     { float hv[];     };
layout(set=0, binding=3)  buffer ZB     { float zb[];     };
layout(set=0, binding=4)  buffer AREA   { float area[];   };
layout(set=0, binding=5)  buffer ELEN   { float elen[];   };
layout(set=0, binding=6)  buffer ENX    { float enx[];    };
layout(set=0, binding=7)  buffer ENY    { float eny[];    };
layout(set=0, binding=8)  buffer ECL    { int   ecL[];    };
layout(set=0, binding=9)  buffer ECR    { int   ecR[];    };
layout(set=0, binding=10) buffer DH     { float dh[];     };
layout(set=0, binding=11) buffer DHU    { float dhu[];    };
layout(set=0, binding=12) buffer DHV    { float dhv[];    };
layout(set=0, binding=13) buffer H1     { float h1[];     };
layout(set=0, binding=14) buffer HU1    { float hu1[];    };
layout(set=0, binding=15) buffer HV1    { float hv1[];    };
layout(set=0, binding=16) buffer NMANN  { float n_mann[]; };
layout(set=0, binding=17) buffer DTBUF  { float dt_buf[]; };
layout(set=0, binding=18) buffer CFLSC  { float cfl_scratch[]; };

layout(push_constant) uniform PC {
    float num_cells;
    float dt;
    float g;
    float dry_tol;
    float cfl_number;
    float stage;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(pc.num_cells)) return;

    float A   = area[i];
    float dt  = pc.dt;
    float dti = dt / A;

    // Stage 0 (predictor): advance from y_n (h/hu/hv)
    // Stage 1 (corrector): advance from y* (h1/hu1/hv1)
    float hi, hui, hvi;
    if (pc.stage < 0.5) {
        hi  = h[i];
        hui = hu[i];
        hvi = hv[i];
    } else {
        hi  = h1[i];
        hui = hu1[i];
        hvi = hv1[i];
    }

    float h_new  = hi  - dti * dh[i];
    float hu_new = hui - dti * dhu[i];
    float hv_new = hvi - dti * dhv[i];

    h_new = max(h_new, 0.0);
    if (h_new < pc.dry_tol) { hu_new = 0.0; hv_new = 0.0; }

    if (h_new >= pc.dry_tol) {
        float n  = n_mann[i];
        float u_ = hu_new / h_new;
        float v_ = hv_new / h_new;
        float spd = sqrt(u_ * u_ + v_ * v_);
        float h43 = pow(h_new, 4.0 / 3.0);
        float Sf  = pc.g * n * n * spd / h43;
        float den = 1.0 + dt * Sf;
        hu_new = hu_new / den;
        hv_new = hv_new / den;
    }

    dh[i]  = 0.0;
    dhu[i] = 0.0;
    dhv[i] = 0.0;

    if (pc.stage < 0.5) {
        h1[i]  = h_new;
        hu1[i] = hu_new;
        hv1[i] = hv_new;
    } else {
        // Heun average: y_{n+1} = 0.5*(y_n + y**)
        // h[i] still holds y_n; h_new = y* + dt*f(y*) = y**
        h[i]  = 0.5 * (h[i]  + h_new);
        hu[i] = 0.5 * (hu[i] + hu_new);
        hv[i] = 0.5 * (hv[i] + hv_new);
    }
}

"""

UPDATE_DTBUF_GLSL = """\
#version 450

layout(local_size_x = 256) in;

layout(set=0, binding=0)  buffer H      { float h[];      };
layout(set=0, binding=1)  buffer HU     { float hu[];     };
layout(set=0, binding=2)  buffer HV     { float hv[];     };
layout(set=0, binding=3)  buffer ZB     { float zb[];     };
layout(set=0, binding=4)  buffer AREA   { float area[];   };
layout(set=0, binding=5)  buffer ELEN   { float elen[];   };
layout(set=0, binding=6)  buffer ENX    { float enx[];    };
layout(set=0, binding=7)  buffer ENY    { float eny[];    };
layout(set=0, binding=8)  buffer ECL    { int   ecL[];    };
layout(set=0, binding=9)  buffer ECR    { int   ecR[];    };
layout(set=0, binding=10) buffer DH     { float dh[];     };
layout(set=0, binding=11) buffer DHU    { float dhu[];    };
layout(set=0, binding=12) buffer DHV    { float dhv[];    };
layout(set=0, binding=13) buffer H1     { float h1[];     };
layout(set=0, binding=14) buffer HU1    { float hu1[];    };
layout(set=0, binding=15) buffer HV1    { float hv1[];    };
layout(set=0, binding=16) buffer NMANN  { float n_mann[]; };
layout(set=0, binding=17) buffer DTBUF  { float dt_buf[]; };
layout(set=0, binding=18) buffer CFLSC  { float cfl_scratch[]; };

layout(push_constant) uniform PC {
    float num_cells;
    float dt_unused;
    float g;
    float dry_tol;
    float cfl_number;
    float stage;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(pc.num_cells)) return;

    float A   = area[i];
    float dt  = dt_buf[0];
    float dti = dt / A;

    float hi, hui, hvi;
    if (pc.stage < 0.5) {
        hi  = h[i];
        hui = hu[i];
        hvi = hv[i];
    } else {
        hi  = h1[i];
        hui = hu1[i];
        hvi = hv1[i];
    }

    float h_new  = hi  - dti * dh[i];
    float hu_new = hui - dti * dhu[i];
    float hv_new = hvi - dti * dhv[i];

    h_new = max(h_new, 0.0);
    if (h_new < pc.dry_tol) { hu_new = 0.0; hv_new = 0.0; }

    if (h_new >= pc.dry_tol) {
        float n  = n_mann[i];
        float u_ = hu_new / h_new;
        float v_ = hv_new / h_new;
        float spd = sqrt(u_ * u_ + v_ * v_);
        float h43 = pow(h_new, 4.0 / 3.0);
        float Sf  = pc.g * n * n * spd / h43;
        float den = 1.0 + dt * Sf;
        hu_new = hu_new / den;
        hv_new = hv_new / den;
    }

    dh[i]  = 0.0;
    dhu[i] = 0.0;
    dhv[i] = 0.0;

    if (pc.stage < 0.5) {
        h1[i]  = h_new;
        hu1[i] = hu_new;
        hv1[i] = hv_new;
    } else {
        h[i]  = 0.5 * (h[i]  + h_new);
        hu[i] = 0.5 * (hu[i] + hu_new);
        hv[i] = 0.5 * (hv[i] + hv_new);
    }
}

"""
