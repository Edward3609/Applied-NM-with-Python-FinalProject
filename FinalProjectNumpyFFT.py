# This version was put through ChatGPT to convert all FFT_CooleyTukey calls to NumPy's faster
#and more robust FFT
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# SAFE INPUT FUNCTIONS
# ==============================

def get_positive_float(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value > 0:
                return value
            else:
                print("❌ Enter a positive number.")
        except:
            print("❌ Invalid input. Enter a number.")

def get_nonzero_float(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value != 0:
                return value
            else:
                print("❌ Value cannot be zero.")
        except:
            print("❌ Invalid input. Enter a number.")


# ==============================
# ABCD MATRICES
# ==============================

def free_space(d):
    return np.array([[1, d], [0, 1]])

def lens(f):
    return np.array([[1, 0], [-1/f, 1]])

def mirror(R):
    return np.array([[1, 0], [-2/R, 1]])

def propagate_q(q, M):
    A, B, C, D = M.flatten()
    return (A*q + B) / (C*q + D)

def beam_radius(q, wavelength):
    return np.sqrt(-wavelength / (np.pi * np.imag(1/q)))

# Curvature from ABCD
def curvature_from_q(q):
    return 1 / np.real(1/q)


# ==============================
# STABILITY
# ==============================

def compute_system_matrix(system):
    M_total = np.eye(2)
    for etype, val in system:
        if etype == "space":
            M = free_space(val)
        elif etype == "lens":
            M = lens(val)
        elif etype == "mirror":
            M = mirror(val)
        M_total = M @ M_total
    return M_total

def check_stability(system):
    M = compute_system_matrix(system)
    A, B, C, D = M.flatten()
    g = (A + D) / 2
    return abs(g) < 1, g

# NEW: count mirrors
def count_mirrors(system):
    return sum(1 for etype, _ in system if etype == "mirror")


# ==============================
# SYSTEM BUILDER
# ==============================

def build_system():
    system = []
    print("\nBuild your optical system:")
    print("Type: space, lens, mirror, done\n")

    while True:
        element = input("Enter element type: ").lower()

        if element == "done":
            if len(system) == 0:
                print("❌ System cannot be empty.")
                continue
            break

        elif element == "space":
            system.append(("space", get_positive_float("Distance (m): ")))

        elif element == "lens":
            system.append(("lens", get_nonzero_float("Focal length (m): ")))

        elif element == "mirror":
            system.append(("mirror", get_nonzero_float("Radius (m): ")))

        else:
            print("❌ Invalid type.")

    return system


# ==============================
# ABCD PROPAGATION
# ==============================

def propagate_system(system, wavelength, q0, dz=0.001):
    z_total, w_total = [], []
    z_current = 0
    q = q0

    for etype, val in system:

        if etype == "space":
            z_vals = np.arange(0, val, dz)

            for z in z_vals:
                q_temp = propagate_q(q, free_space(z))
                w = beam_radius(q_temp, wavelength)

                z_total.append(z_current + z)
                w_total.append(w)

            q = propagate_q(q, free_space(val))
            z_current += val

        elif etype == "lens":
            q = propagate_q(q, lens(val))

        elif etype == "mirror":
            q = propagate_q(q, mirror(val))

    return np.array(z_total), np.array(w_total), q


# ==============================
# GAUSSIAN FIELD
# ==============================

def gaussian_beam_2D(N, dx, w0):
    x = np.linspace(-N/2*dx, N/2*dx, N)
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / w0**2).astype(complex)


# ==============================
# FFT PROPAGATION
# ==============================

def fresnel_propagation(U0, dx, wavelength, z):
    N = U0.shape[0]
    fx = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fx)

    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    return np.fft.ifft2(np.fft.fft2(U0) * H)


def propagate_fft_final(system, U0, dx, wavelength):
    U = U0.copy()

    for etype, val in system:

        if etype == "space":
            U = fresnel_propagation(U, dx, wavelength, val)

        elif etype == "lens":
            N = U.shape[0]
            x = np.linspace(-N/2*dx, N/2*dx, N)
            X, Y = np.meshgrid(x, x)
            U *= np.exp(-1j * np.pi/(wavelength*val)*(X**2 + Y**2))

        elif etype == "mirror":
            N = U.shape[0]
            x = np.linspace(-N/2*dx, N/2*dx, N)
            X, Y = np.meshgrid(x, x)
            U *= np.exp(-1j * 2*np.pi/(wavelength*val)*(X**2 + Y**2))

    return U


# ==============================
# BEAM WIDTH
# ==============================

def compute_beam_width(I, dx):
    N = I.shape[0]
    x = np.linspace(-N/2*dx, N/2*dx, N)
    X, Y = np.meshgrid(x, x)

    r2 = X**2 + Y**2
    return np.sqrt(2 * np.sum(r2 * I) / np.sum(I))


# ==============================
# POWER
# ==============================

def compute_power(U):
    return np.sum(np.abs(U)**2)


# ==============================
# FFT CURVATURE
# ==============================

def extract_wavefront_curvature(U, dx, wavelength):
    N = U.shape[0]
    x = np.linspace(-N/2*dx, N/2*dx, N)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2

    I = np.abs(U)**2

    phase = np.angle(U)
    phase = np.unwrap(np.unwrap(phase, axis=0), axis=1)

    r2_flat = r2.flatten()
    phase_flat = phase.flatten()
    I_flat = I.flatten()

    mask = I_flat > 0.2 * np.max(I_flat)

    r2_fit = r2_flat[mask]
    phase_fit = phase_flat[mask]

    A = np.vstack([r2_fit, np.ones_like(r2_fit)]).T
    a, _ = np.linalg.lstsq(A, phase_fit, rcond=None)[0]

    k = 2*np.pi / wavelength
    R = k / (2*a)

    return R


# ==============================
# COLOR
# ==============================

def wavelength_to_rgb(wavelength):
    wl = wavelength * 1e9
    if wl < 490:
        return np.array([0,0,1])
    elif wl < 580:
        return np.array([0,1,0])
    else:
        return np.array([1,0,0])


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    w0 = get_positive_float("Beam waist w0 (m): ")
    wavelength = get_positive_float("Wavelength (m): ")

    q0 = 1j * np.pi * w0**2 / wavelength
    system = build_system()

    # ===== STABILITY (ONLY IF CAVITY) =====
    print("\n===== STABILITY CHECK =====")
    num_mirrors = count_mirrors(system)

    if num_mirrors >= 2:
        is_stable, g = check_stability(system)
        print(f"Mirrors detected: {num_mirrors} → Resonator system")
        print(f"(A + D)/2 = {g:.4f}")
        print("✅ STABLE CAVITY" if is_stable else "❌ UNSTABLE CAVITY")
    else:
        print(f"Mirrors detected: {num_mirrors}")
        print("ℹ️ Stability not applicable (not a resonator)")

    # ===== ABCD =====
    z, w, q_final = propagate_system(system, wavelength, q0)

    rgb = wavelength_to_rgb(wavelength)

    plt.figure()
    plt.plot(z, w, color=rgb)
    plt.plot(z, -w, color=rgb)
    plt.fill_between(z, -w, w, color=rgb, alpha=0.2)
    plt.title("ABCD Beam Envelope")

    # ===== FFT =====
    N = 2048
    L = 1e-2
    dx = L / N

    U0 = gaussian_beam_2D(N, dx, w0)

    P_initial = compute_power(U0)

    U = propagate_fft_final(system, U0, dx, wavelength)

    P_final = compute_power(U)

    I = np.abs(U)**2
    I /= np.max(I)

    # ===== WIDTH =====
    w_fft = compute_beam_width(I, dx)
    w_abcd = w[-1]
    width_error = abs(w_fft - w_abcd) / w_abcd * 100

    # ===== CURVATURE =====
    R_fft = extract_wavefront_curvature(U, dx, wavelength)
    R_abcd = curvature_from_q(q_final)
    curvature_error = abs(R_fft - R_abcd) / abs(R_abcd) * 100

    # ===== POWER =====
    power_error = abs(P_final - P_initial) / P_initial * 100

    print("\n===== FINAL BEAM COMPARISON =====")
    print(f"FFT Beam Radius   = {w_fft:.6f} m")
    print(f"ABCD Beam Radius  = {w_abcd:.6f} m")
    print(f"Percent Error     = {width_error:.2f}%")

    print("\n===== CURVATURE COMPARISON =====")
    print(f"FFT Curvature R   = {R_fft:.6f} m")
    print(f"ABCD Curvature R  = {R_abcd:.6f} m")
    print(f"Curvature Error   = {curvature_error:.2f}%")

    print("\n===== POWER CONSERVATION =====")
    print(f"Initial Power     = {P_initial:.6e}")
    print(f"Final Power       = {P_final:.6e}")
    print(f"Power Error       = {power_error:.4f}%")

    # ===== COLOR PLOT =====
    scale = 3
    x_limit = scale * w_fft

    I_rgb = np.zeros((I.shape[0], I.shape[1], 3))
    for i in range(3):
        I_rgb[:,:,i] = I * rgb[i]

    plt.figure()
    plt.imshow(I_rgb, extent=[-x_limit, x_limit, -x_limit, x_limit])
    plt.title("Final Beam Intensity (Auto-Scaled)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.show()
