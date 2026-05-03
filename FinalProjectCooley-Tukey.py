import numpy as np
import matplotlib.pyplot as plt

# ==============================
# SAFE INPUT
# ==============================

def get_positive_float(prompt):
    while True:
        try:
            v = float(input(prompt))
            if v > 0:
                return v
            print("Enter positive number")
        except:
            print("Invalid input")

def get_nonzero_float(prompt):
    while True:
        try:
            v = float(input(prompt))
            if v != 0:
                return v
            print("Cannot be zero")
        except:
            print("Invalid input")


# ==============================
# COOLEY–TUKEY FFT
# ==============================

def fft_cooley_tukey(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    if N == 1:
        return x
    even = fft_cooley_tukey(x[::2])
    odd = fft_cooley_tukey(x[1::2])
    factor = np.exp(-2j*np.pi*np.arange(N)/N)
    return np.concatenate([
        even + factor[:N//2]*odd,
        even - factor[:N//2]*odd
    ])

def ifft_cooley_tukey(x):
    return np.conj(fft_cooley_tukey(np.conj(x))) / len(x)

def fft2_cooley_tukey(U):
    temp = np.array([fft_cooley_tukey(row) for row in U])
    return np.array([fft_cooley_tukey(col) for col in temp.T]).T

def ifft2_cooley_tukey(U):
    temp = np.array([ifft_cooley_tukey(row) for row in U])
    return np.array([ifft_cooley_tukey(col) for col in temp.T]).T


# ==============================
# ABCD
# ==============================

def free_space(d):
    return np.array([[1,d],[0,1]])

def lens(f):
    return np.array([[1,0],[-1/f,1]])

def mirror(R):
    return np.array([[1,0],[-2/R,1]])

def propagate_q(q, M):
    A,B,C,D = M.flatten()
    return (A*q + B)/(C*q + D)

def beam_radius(q, wavelength):
    return np.sqrt(-wavelength/(np.pi*np.imag(1/q)))

def curvature_from_q(q):
    return 1/np.real(1/q)


# ==============================
# STABILITY
# ==============================

def compute_system_matrix(system):
    M = np.eye(2)
    for etype,val in system:
        if etype=="space": mat=free_space(val)
        elif etype=="lens": mat=lens(val)
        elif etype=="mirror": mat=mirror(val)
        M = mat @ M
    return M

def check_stability(system):
    M = compute_system_matrix(system)
    A,B,C,D = M.flatten()
    g = (A+D)/2
    return abs(g)<1, g

def count_mirrors(system):
    return sum(1 for e,_ in system if e=="mirror")


# ==============================
# SYSTEM BUILDER
# ==============================

def build_system():
    system=[]
    print("\nBuild system: space, lens, mirror, done\n")
    while True:
        e=input("Element: ").lower()
        if e=="done": break
        elif e=="space":
            system.append(("space", get_positive_float("Distance: ")))
        elif e=="lens":
            system.append(("lens", get_nonzero_float("Focal length: ")))
        elif e=="mirror":
            system.append(("mirror", get_nonzero_float("Radius: ")))
        else:
            print("Invalid")
    return system


# ==============================
# ABCD PROPAGATION
# ==============================

def propagate_system(system, wavelength, q0, dz=0.002):
    z_total=[]; w_total=[]
    z=0; q=q0

    for etype,val in system:
        if etype=="space":
            z_vals=np.arange(0,val,dz)
            for dz_i in z_vals:
                q_temp=propagate_q(q, free_space(dz_i))
                w=beam_radius(q_temp, wavelength)
                z_total.append(z+dz_i)
                w_total.append(w)
            q=propagate_q(q, free_space(val))
            z+=val
        elif etype=="lens":
            q=propagate_q(q, lens(val))
        elif etype=="mirror":
            q=propagate_q(q, mirror(val))

    return np.array(z_total), np.array(w_total), q


# ==============================
# GAUSSIAN
# ==============================

def gaussian_beam_2D(N, dx, w0):
    x=np.linspace(-N/2*dx,N/2*dx,N)
    X,Y=np.meshgrid(x,x)
    return np.exp(-(X**2+Y**2)/w0**2).astype(complex)


# ==============================
# FFT PROPAGATION
# ==============================

def fresnel_propagation(U0, dx, wavelength, z):
    N=U0.shape[0]
    fx=np.fft.fftfreq(N,dx)
    FX,FY=np.meshgrid(fx,fx)
    H=np.exp(-1j*np.pi*wavelength*z*(FX**2+FY**2))
    return ifft2_cooley_tukey(fft2_cooley_tukey(U0)*H)

def propagate_fft_system(system, U0, dx, wavelength):
    U=U0.copy()

    for etype,val in system:
        if etype=="space":
            U=fresnel_propagation(U,dx,wavelength,val)
        elif etype=="lens":
            N=U.shape[0]
            x=np.linspace(-N/2*dx,N/2*dx,N)
            X,Y=np.meshgrid(x,x)
            U*=np.exp(-1j*np.pi/(wavelength*val)*(X**2+Y**2))
        elif etype=="mirror":
            N=U.shape[0]
            x=np.linspace(-N/2*dx,N/2*dx,N)
            X,Y=np.meshgrid(x,x)
            U*=np.exp(-1j*2*np.pi/(wavelength*val)*(X**2+Y**2))

    return U


# ==============================
# METRICS
# ==============================

def compute_beam_width(I, dx):
    N=I.shape[0]
    x=np.linspace(-N/2*dx,N/2*dx,N)
    X,Y=np.meshgrid(x,x)
    I=I/np.sum(I)
    r2=X**2+Y**2
    return np.sqrt(2*np.sum(r2*I))

def compute_power(U):
    return np.sum(np.abs(U)**2)


# ==============================
# CURVATURE
# ==============================

def extract_wavefront_curvature(U, dx, wavelength):
    N=U.shape[0]
    x=np.linspace(-N/2*dx,N/2*dx,N)
    X,Y=np.meshgrid(x,x)
    r2=X**2+Y**2

    I=np.abs(U)**2
    phase=np.unwrap(np.unwrap(np.angle(U),axis=0),axis=1)

    r2_flat=r2.flatten()
    phase_flat=phase.flatten()
    I_flat=I.flatten()

    mask=I_flat>0.2*np.max(I_flat)

    A=np.vstack([r2_flat[mask],np.ones_like(r2_flat[mask])]).T
    a,_=np.linalg.lstsq(A,phase_flat[mask],rcond=None)[0]

    k=2*np.pi/wavelength
    return k/(2*a)


# ==============================
# COLOR
# ==============================

def wavelength_to_rgb(wavelength):
    wl=wavelength*1e9
    if wl<490: return np.array([0,0,1])
    elif wl<580: return np.array([0,1,0])
    else: return np.array([1,0,0])


# ==============================
# MAIN
# ==============================

if __name__=="__main__":

    w0=get_positive_float("Beam waist w0: ")
    wavelength=get_positive_float("Wavelength: ")

    rgb=wavelength_to_rgb(wavelength)

    q0=1j*np.pi*w0**2/wavelength
    system=build_system()

    print("\n===== STABILITY CHECK =====")
    m=count_mirrors(system)
    if m>=2:
        stable,g=check_stability(system)
        print(f"(A + D)/2 = {g:.4f}")
        print("✅ STABLE CAVITY" if stable else "❌ UNSTABLE CAVITY")
    else:
        print("ℹ️ Not a resonator")

    z,w,q_final=propagate_system(system,wavelength,q0)

    plt.figure()
    plt.plot(z,w,color=rgb)
    plt.plot(z,-w,color=rgb)
    for a in np.linspace(0.05,0.25,5):
        plt.fill_between(z,-w,w,color=rgb,alpha=a)
    plt.title("Beam Envelope")
    plt.grid()

    N=1024
    L=8e-3
    dx=L/N

    U0=gaussian_beam_2D(N,dx,w0)
    P0=compute_power(U0)

    U=propagate_fft_system(system,U0,dx,wavelength)
    Pf=compute_power(U)

    I=np.abs(U)**2
    w_fft=compute_beam_width(I,dx)
    R_fft=extract_wavefront_curvature(U,dx,wavelength)

    w_abcd=w[-1]
    R_abcd=curvature_from_q(q_final)

    width_error = abs(w_fft - w_abcd) / w_abcd * 100
    curvature_error = abs(R_fft - R_abcd) / abs(R_abcd) * 100
    power_error = abs(Pf - P0) / P0 * 100

    print("\n===== FINAL BEAM COMPARISON =====")
    print(f"FFT Beam Radius   = {w_fft:.6f} m")
    print(f"ABCD Beam Radius  = {w_abcd:.6f} m")
    print(f"Percent Error     = {width_error:.2f}%")

    print("\n===== CURVATURE COMPARISON =====")
    print(f"FFT Curvature R   = {R_fft:.6f} m")
    print(f"ABCD Curvature R  = {R_abcd:.6f} m")
    print(f"Curvature Error   = {curvature_error:.2f}%")

    print("\n===== POWER CONSERVATION =====")
    print(f"Initial Power     = {P0:.6e}")
    print(f"Final Power       = {Pf:.6e}")
    print(f"Power Error       = {power_error:.4f}%")

    I/=np.max(I)
    I_rgb=np.zeros((N,N,3))
    for i in range(3):
        I_rgb[:,:,i]=I*rgb[i]

    plt.figure()
    plt.imshow(I_rgb)
    plt.title("Final Beam (Colored)")
    plt.show()