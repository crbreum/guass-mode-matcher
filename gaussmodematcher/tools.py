import time
import numpy as np
import sympy as sym
from scipy.optimize import fsolve, least_squares
from scipy import special
from itertools import combinations, permutations
import matplotlib.pyplot as plt
from matplotlib import lines
import lmfit as lmf
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import IPython.display as disp

c_speed = 299792458
#path_fig = 'C:/Users/crbr/Dropbox/Postdoc/projects/introbox/figures/'



class beam():
    def __init__(self, w0, z0=0, n0=1, lam=1550e-6):
        self.w0 = w0
        self.z0 = z0
        self.n0 = n0
        self.lam = lam
        self.zR = n0*np.pi*w0**2 / lam
        self.divergence = lam / (np.pi*n0*w0)
        self.paraxial_limit = w0 / lam
        self.NA = lam/(np.pi*w0)
    
    def w(self,x):
        z = x-self.z0
        w = self.w0 * np.sqrt(1 + (z/self.zR)**2)
        return w
    
    def R(self, x):
        z = x-self.z0
        R = z*(1 + (self.zR/z)**2)
        return R
    
    def q(self, z):
        return z + 1j*self.zR
    
    def Gouy_phase(self, x):
        z = x - self.z0
        psi = np.arctan(z/self.zR)
        return psi
    
    def intensity(self,x,y,z,P0=1):
        r2 = x**2 + y**2
        I0 = 2*P0/(np.pi*self.w0**2) 
        return I0*(self.w0/self.w(z))**2 * np.exp(-2*r2/self.w(z)**2)
    
    def cross(self, z=0, P0=1, r=1, res=101, steps=False):
        x,dx = np.linspace(-r,r,res, retstep=True)
        y,dy = np.linspace(-r,r,res, retstep=True)
        xs,ys = np.meshgrid(x,y)
        if steps:
            return self.intensity(xs,ys,z,P0), xs, ys
        else:
            return self.intensity(xs,ys,z,P0)
    
    def simdata_crosscut1(self, z, dx=0.1, r=1, res=101):
        dxs = np.arange(-r,r,dx)
        def crosscut(x):
            x,dx = np.linspace(-r,-x,res, retstep=True)
            y,dy = np.linspace(-r,r,res, retstep=True)
            xs,ys = np.meshgrid(x,y)
            return np.sum(self.intensity(xs,ys,z)*dx*dy)
        return np.array([crosscut(x) for x in dxs])
    
    def simdata_crosscut2(self, z, dx=0.1, r=1):
        dxs = np.arange(-r,r,dx)
        def func(z,x):
            return 1/2*special.erfc(np.sqrt(2)*x/self.w(z))
        return np.array([func(z,x) for x in dxs])
        
        

    
    
class beamplots():
    def __init__(self, lam=1550e-6):
        self.lam = lam
        
    def plot(self,w0,n):
        w0max = 0.1**(4-n)
        bmax = beam(w0max, 0, 1, self.lam)
        b = beam((w0/10)*w0max, 0, 1, self.lam)
        z = 3.5*np.linspace(-bmax.zR,bmax.zR,100) 
        z2 = 3.5*np.linspace(0,bmax.zR,50)
        z3 = np.linspace(-b.zR,b.zR,50)

        fig,ax = plt.subplots(figsize=(12,8))

        def phasefront2_xy(z,side):
            r = b.R(z)
            y = np.linspace(-b.w(z),b.w(z))
            x = side*np.sqrt(np.abs(r**2 - y**2)) + side*(z - r)
            return x,y
        for i in [1,2,3]:
            plt.plot(*phasefront2_xy(i*b.zR,1), 'C3', alpha=.3)
            plt.plot(*phasefront2_xy(i*b.zR,-1), 'C3', alpha=.3)

        ax.plot(z, b.w(z), color='C3')
        ax.plot(z, -b.w(z), color='C3')
        ax.fill_between(z3,-b.w(z3),b.w(z3), color='C3', alpha=0.05)

        ax.plot(z2, z2*np.tan(b.divergence), 'k--', alpha=0.5)
        phi2 = np.linspace(0,b.divergence,50) / (2*b.w0/b.zR)
        r = b.zR/2
        x2 = r*np.cos(phi2)
        y2 = r*np.sin(phi2) * 2*b.w0/b.zR
        ax.plot(x2,y2, 'k', alpha=.5)
        plt.text(b.zR/2,0.2*b.w0, ' $\\theta$ ', ha='left')

        ax.annotate(text='z', xy=(-3*bmax.zR,0), xytext=(3*bmax.zR,0), arrowprops=dict(arrowstyle='<-'), va='center')

        ax.annotate(text='', xy=(0,0), xytext=(0,b.w0), arrowprops=dict(arrowstyle='<->'), ha='center')
        t1 = plt.text(0,b.w0/2, '$\omega_0$ ', ha='right')

        ax.annotate(text='', xy=(-b.zR,0), xytext=(-b.zR,b.w(b.zR)), arrowprops=dict(arrowstyle='<->'))
        t2 = plt.text(-b.zR,b.w0/2, '$\sqrt{2}\omega_0$ ', ha='right')

        ax.annotate(text='', xy=(-b.zR,b.w(b.zR)*1.1), xytext=(b.zR,b.w(b.zR)*1.1), arrowprops=dict(arrowstyle='<->'))
        t2 = plt.text(0,b.w(b.zR)*1.2, '$2z_R$', ha='center')


        #t = 0.0005*np.linspace(-np.pi,np.pi,50)
        def phasefront_xy(z):
            r = b.R(z)
            phi = np.arctan(b.w(z)/z)
            phis = np.linspace(-phi,phi,50) /2 # (2*b.w0/b.zR)
            x = r*(np.cos(phis)-1) + z
            y = r*np.sin(phis)
            return x,y    

        text_info = ('waist:   '+r'$\omega_0=$'+f'{b.w0*1e3 : .2f}um \n'
                    +'Rayleigh length:   '+r'$z_R=$'+f'{b.zR : .3f}mm \n'
                    +'Maximum curve:   '+r'$R(z_R)=$'+f'{b.R(b.zR) : .3f}mm\n'
                    +'Divergence angle:   '+r'$\theta=$'+f'{b.divergence*180/np.pi : .2f} deg. \n'
                    +'Paraxial limit:   '+r'$\omega_0 / \lambda=$'+f'{b.paraxial_limit : .2f}\n'
                    +'Numerical apperature:   '+r'NA$=$'+f'{b.NA : .4f}\n'
                    )
        ax.text(bmax.zR*3.7,0, text_info)

        ax.grid(alpha=0.5)
        ax.set_aspect(bmax.zR/(2*bmax.w0))
        plt.xlim(-3.5*bmax.zR,3.5*bmax.zR)
        plt.ylim(-bmax.w(z[-1]),bmax.w(z[-1]))
        plt.show()

    
    
    
    
class modematching():
    def __init__(self, lam=1550e-6):
        self.lam = lam
    
    def w(self,z, w0, n0=1):
        zR = zR = n0*np.pi*w0**2/self.lam
        w = w0 * np.sqrt(1 + (z/zR)**2)
        return w
    
    def cross_fit(self, x, data, guess=[0,1], sub_offset=False, plot=False):        
        def model(x,x0,w,offset=0):
            return 1/2 * special.erfc(np.sqrt(2)*(x-x0)/w) + offset
        
        def residual(params, x, data, offset=0):
            x0 = params['x0']
            w = params['w']
            return data-model(x,x0,w,offset)
        
        params = lmf.Parameters()
        params.add('x0', value=guess[0], min=-1, max=1, vary=True)
        params.add('w', value=guess[1], min=1e-4, max=1e2, vary=True)
        if sub_offset:
            tail = np.int(len(data)*3/4)
            offset = data[tail:].mean()
        else:
            offset=0
        out = lmf.minimize(residual, params, args=(x, data, offset))
        
        out_values = []
        out_stderr = []
        for key in params.keys():
            out_values = np.append(out_values, out.params[key].value)
            out_stderr = np.append(out_stderr, out.params[key].stderr)
            
        x0 = out_values[0]
        w = out_values[1]
        fitx = np.linspace(x[0],x[-1],100)
        fity = model(fitx, *out_values)+offset
        
        if plot:
            fig, ax = plt.subplots()
            ax.plot(x, data, 'o', label='data')
            ax.plot(fitx, fity, 'r', label='fit to erfc(x):\n'+r'$\omega=${:.2f}$\pm${:.2f}mm'.format(w, out_stderr[1]))
            ax.set_xlabel('Cross-section axis: x [mm]')
            ax.set_ylabel('Relative power')
            ax.legend()
            plt.grid(alpha=0.5)
            plt.show()           
        
        return x0,w,fitx,fity
    
    def beam_fit(self,x, data, guess=[0,1e-1], plot=False):       
        def model(z,z0,w0):
            return self.w(z-z0,w0)
        
        def residual(params, x, data):
            z0 = params['z0']
            w0 = params['w0']
            return data-model(x,z0,w0)
        
        params = lmf.Parameters()
        params.add('z0', value=guess[0], min=-1e3, max=1e3, vary=True)
        params.add('w0', value=guess[1], min=1e-4, max=1e2, vary=True)
        out = lmf.minimize(residual, params, args=(x, data))
        
        out_values = []
        out_stderr = []
        for key in params.keys():
            out_values = np.append(out_values, out.params[key].value)
            out_stderr = np.append(out_stderr, out.params[key].stderr)
        
        z0 = out_values[0]
        w0 = out_values[1]
        x0 = np.abs(x[[0,-1]]-z0).max()
        z = np.linspace((z0-x0)*1.1,(z0+x0)*1.1,100)
        
        # Linear fit
        i_under = [i for i,k in enumerate(x) if k < z0]
        i_over = [i for i,k in enumerate(x) if k > z0]
        if np.argmax([len(i_under), len(i_over)]) == 0:
            z_linear = x[i_under]
            w_linear = data[i_under]
        else:
            z_linear = x[i_over]
            w_linear = data[i_over]
            
        f_linear = lambda x,a,x0: a*(x-x0)
        model2 = lmf.Model(f_linear, independent_vars=['x'])
        a_guess = w_linear[-1]/(z_linear[-1]-z0)
        out2 = model2.fit(w_linear, x=z_linear, a=a_guess, x0=z0)   
        
        out2_values = []
        out2_stderr = []
        for key in ['a', 'x0']:
            out2_values = np.append(out2_values, out2.params[key].value)
            out2_stderr = np.append(out2_stderr, out2.params[key].stderr)
        
        
        theta = np.arctan(out2_values[0])
        theta_stderr = np.arctan(out2_stderr[0])
        M2 = (self.lam/np.pi) / (w0*theta)
        M2_stderr = theta*out_stderr[1] + w0*theta_stderr
        z_linear2 = np.linspace(z0, z_linear[-1]*1.1,100)
        
        if plot:
            fit_label1 = r'$z_0=${:.0f}$\pm${:.1f}mm'.format(z0, out_stderr[0]) + '\n' + r'$w_0=${:.0f}$\pm${:.1f}um'.format(w0*1e3, out_stderr[1]*1e3)
            fit_label2 = ('linear fit: '+r'$\theta=${:.1f}$\pm${:.1f}'.format(theta*180/np.pi, theta_stderr*180/np.pi)+'\n'+ 
                         r'$M^2=${:.2f}$\pm${:.3f}'.format(M2, M2_stderr))

            fig, ax = plt.subplots()
            ax.plot(x, data, 'o', label='data')
            ax.plot(z, model(z, *out_values), 'r', label=r'fit to $\omega(z)$:')
            ax.plot(z0, w0, 'kx', label=fit_label1)
            ax.plot(z_linear2, f_linear(z_linear2, *out2_values), 'k--', alpha=0.5, label=fit_label2)
            ax.set_xlabel('Optical axis, z [mm]')
            ax.set_ylabel('Beam width, w(z) [mm]')
            ax.legend()
            plt.show()
        return out_values, out_stderr
    
    def match_df(self, w01,d1,f1 ,plot=False):
        # Function to find position and size of new waist (w02) after placing 1 lense
        # pams = [distance from w01 to lense, f-number of lense]      
        n, lam = 1, self.lam
        zR1 = np.pi*n*w01**2/lam

        d2 = np.abs(f1*(d1**2 - d1*f1 + zR1**2)/(d1**2 - 2*d1*f1 + f1**2 + zR1**2))
        zR2 = f1**2*zR1/np.abs(d1**2 - 2*d1*f1 + f1**2 + zR1**2)
        w02 = np.sqrt(lam*zR2/np.pi)
        
        if plot:
            z02 = d1+d2

            b1 = beam(w01,0)
            z1 = np.linspace(0,d1,100)
            p1 = b1.w(z1)

            b2 = beam(w02, z02)
            z2 = np.linspace(d1,z02*1.2,100)
            p2 = b2.w(z2)

            plt.plot(z1,p1, 'C0')
            plt.plot(z2,p2, 'C0')
            plt.grid()
        
        return w02, d2  
        

    def match_1f(self, pams, lenses, plot=False):
        # pams = [w01, w02, z02]
        def solver_1f(ds, pams, fs):
            d1,d2 = ds[0], ds[1]

            n,lam = 1, self.lam
            w01,w02 = pams[0], pams[1]
            zR1, zR2  = np.pi*n*w01**2/lam, np.pi*n*w02**2/lam
            f1 = fs

            M = 1/f1 * np.matrix([[-d2 + f1, -d1*d2 + (d1+d2)*f1], [-1, -d1+f1]])
            _A,_B,_C,_D = M[0,0], M[0,1], M[1,0], M[1,1]

            F = np.empty((2))
            F[0] = -_B*_D/(_A*_C) - zR1**2
            F[1] = -_A*_B/(_C*_D) - zR2**2 
            return F
        
        def lens_placement_check(z, pam):
            check1 = all(x > 0 for x in z)
            check_all = check1
            return check_all
        
        print('-=[ Solutions ]=-')
        for i,fs in enumerate(lenses):
            ier = 2
            n_try = 0
            lens_check = False
            z0 = np.array([10,10])
            while ier > 1 and n_try < 10 or lens_check == False:
                z,out,ier,mesg = fsolve(solver_1f, z0, args=(pams, fs), full_output=True)
                z0 += np.array([50,50])
                lens_check = lens_placement_check(z, pams)
                n_try += 1                
            if ier == 1:
                print(f'f{fs}: [{z[0]:.1f}, {z[0]+z[1]:.1f}]mm')  
            else:
                print(f'f{fs}: None')
                
            if plot:
                w01,w02 = pams[0], pams[1]
                d1,d2 = z[0], z[1]
                z02 = d1+d2

                b1 = beam(w01,0)
                z1 = np.linspace(0,d1,100)
                p1 = b1.w(z1)

                b2 = beam(w02, z02)
                z2 = np.linspace(d1,z02*1.2,100)
                p2 = b2.w(z2)

                plt.plot(z1,p1, f'C{i:.0f}', label=f'f{fs}')
                plt.plot(z2,p2, f'C{i:.0f}')
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()
                      
                      
                

    def match_2f(self, pams, lenses, plot=False):        
        def solver_2f(ds, pams, fs):
            d1,d2,d3 = ds[0], ds[1], ds[2]

            n,lam = 1, self.lam
            w01,w02 = pams[0], pams[1]
            zR1, zR2  = np.pi*n*w01**2/lam, np.pi*n*w02**2/lam
            d = pams[2]
            f1,f2 = fs[0], fs[1]

            k1 = d2*(d3-f2)-d3*(f1+f2)+f1*f2
            k2 = d2-f1-f2
            k3 = -d2*f1 + f1*f2
            M = 1/(f1*f2)*np.matrix([[k1, d1*k1 + d2*f1*f2 + d3*k3], [k2, d1*k2 + k3]])
            _A,_B,_C,_D = M[0,0], M[0,1], M[1,0], M[1,1]

            F = np.empty((3))
            F[0] = -_B*_D/(_A*_C) - zR1**2
            F[1] = -_A*_B/(_C*_D) - zR2**2 
            F[2] = d1 + d2 + d3 - d
            #F[2] = np.heaviside(d1,0)*d1 + np.heaviside(d2,0)*d2 + np.heaviside(d3,0)*d3 - d
            return F
        
        def build_lenscombination(lenses):
            comb_all = list(permutations(lenses, 2))
            comb_uniq = [i for n, i in enumerate(comb_all) if i not in comb_all[:n]]
            return comb_uniq
        
        def build_z0(d, dz=50, dlim=25):
            f1s = []
            f2s = []
            f1z = np.arange(dlim, d-dlim, dz)
            for i1,k1 in enumerate(f1z):
                f2z=np.arange(k1+dlim, d, dz)
                for i2,k2 in enumerate(f2z):
                    f1s = np.append(f1s,k1)
                    f2s = np.append(f2s,k2)
            z0_guess = np.column_stack((f1s, f2s))
            return z0_guess
        
        def lens_placement_check(z, pam):
            check1 = all(x > 0 for x in z)
            check2 = sum(z) <= pam[-1] 
            check_all = check1 and check2
            return check_all
        
        # Code
        lenslist = build_lenscombination(lenses)
        z0list = build_z0(pams[-1], 49, 19)
        ic = 0
        print('-=[ Solutions ]=-')
        for i,fs in enumerate(lenslist):
            zout_all = np.zeros(3)
            for ik,z0f in enumerate(z0list):
                z0z = np.append(z0f, pams[-1]-z0f.sum())
                z,out,ier,mesg = fsolve(solver_2f, z0z, args=(pams, fs), full_output=True)
                lens_check = lens_placement_check(z, pams)
                if ier == 1 and lens_check == True:
                    zout_all=np.row_stack((zout_all,np.round(z,3)))
            zouts = [n for ik, n in enumerate(zout_all) if n not in zout_all[:ik]]
            zouts = np.delete(zouts,0,0)
            #print(fs,zout_all)
            if len(zouts) == 0:
                print(f'f{fs}: None')
            elif len(zouts) == 1:
                string_zouts = f'[{zouts[0,0]:.1f}, {zouts[0,0:2].sum():.1f}]mm'
                print(f'f{fs}:', string_zouts)
            elif len(zouts == 2):
                string_zouts = f'[{zouts[0,0]:.1f}, {zouts[0,0:2].sum():.1f}]mm, [{zouts[1,0]:.1f}, {zouts[1,0:2].sum():.1f}]mm'
                print(f'f{fs}:', string_zouts)
            
            
            if plot and len(zouts) > 0:
                for i_ls, zout in enumerate(zouts):
                    w01,w03,z03 = pams[0], pams[1], pams[2]
                    d1,d2,d3 = zout[0], zout[1], zout[2]

                    w02,z02 =  self.match_df(w01,d1,fs[0])
                    z02 += d1

                    b1 = beam(w01,0)
                    z1 = np.linspace(0,d1,100)
                    p1 = b1.w(z1)

                    b2 = beam(w02, z02)
                    z2 = np.linspace(d1,d1+d2,100)
                    p2 = b2.w(z2)

                    b3 = beam(w03, z03)
                    z3 = np.linspace(d1+d2,z03*1.2,100)
                    p3 = b3.w(z3)
                    
                    plt.plot(z1,p1, f'C{ic:.0f}', label=f'f{fs}', linestyle=list(lines.lineStyles.keys())[i_ls])
                    plt.plot(z2,p2, f'C{ic:.0f}', linestyle=list(lines.lineStyles.keys())[i_ls])
                    plt.plot(z3,p3, f'C{ic:.0f}', linestyle=list(lines.lineStyles.keys())[i_ls])
                ic +=1
        plt.legend()
        plt.grid(alpha=.5)
        plt.show()
        
        

class abcd():
    def __init__(self):
        M1 = lambda d: np.matrix([[1, d], [0, 1]])
        M2 = lambda f: np.matrix([[1, 0], [-1/f, 1]])
        M3 = lambda r: np.matrix([[1, 0], [-2/r, 1]])
        M4 = lambda n1,n2: np.matrix([[1, 0], [0, n1/n2]])
        M5 = lambda n1,n2,r: np.matrix([[1, 0], [(n1-n2)/(n2*r), n1/n2]])
        self.lam = lam
    
    def prop(d):
        return np.matrix([[1, d], [0, 1]])
    
    def lens(f):
        return np.matrix([[1, 0], [-1/f, 1]])
    
    def curv(R,theta=0, astig='none'):
        if astig == 'none':
            Rd = R
        if astig == 'v':
            Rd = R/np.cos(theta)
        elif astig == 'h':
            Rd = R*np.cos(theta)
            
        return np.matrix([[1, 0], [-2/Rd, 1]])
    
    def infa(n1,n2):
        return np.matrix([[1, 0], [0, n1/n2]])
    
    def cuinfa(n1,n2,R):
        return np.matrix([[1, 0], [(n1-n2)/(n2*R), n1/n2]])
    
    def overlap(w1,w2):
        return 2*(w2/w1+w1/w2)**(-1)
    
    def w2w(M,n1=1,n2=1,lam=1550e-6):
        _A = M[0,0]
        _B = M[0,1]
        _C = M[1,0]
        _D = M[1,1]
        
        if (-_B*_D)/(_A*_C) < 0:
            w1 = None
        else:
            w1 = np.sqrt(lam/(n1*np.pi)*np.sqrt(-_B*_D/(_A*_C)))
        
        if (-_A*_B/(_C*_D)) < 0:
            w2 = None
        else:
            w2 = np.sqrt(lam/(n2*np.pi)*np.sqrt(-_A*_B/(_C*_D)))
            
        return w1, w2
    
class abcd_sym():
    def __init__(self):
        pass
    def prop(d):
        return sym.Matrix([[1, d], [0, 1]])
    
    def lens(f):
        return sym.Matrix([[1, 0], [-1/f, 1]])
    
    def curv(R,theta=0, astig='none'):
        if astig == 'none':
            Rd = R
        if astig == 'v':
            Rd = R/sym.cos(theta)
        elif astig == 'h':
            Rd = R*sym.cos(theta)
            
        return sym.Matrix([[1, 0], [-2/Rd, 1]])
    
    def infa(n1,n2):
        return sym.Matrix([[1, 0], [0, n1/n2]])
    
    def cuinfa(n1,n2,R):
        return sym.Matrix([[1, 0], [(n1-n2)/(n2*R), n1/n2]])
        
def sellmeier(lam, *coefficients):
    """Sellmeier equation for refractive index at 
       wavelength _lam_ (um) for crystal with given coefficients."""
    c0, c1, c2, c3, c4 = coefficients
    return sqrt(c0 + c1/(lam**2 - c2) + c3/(lam**2 - c4))

def deltan_coeff(lam, *coefficients):
    """3rd order polynomial coefficients of the 2nd order
       temperature correction polynomial."""
    a0, a1, a2, a3 = coefficients
    return a0 + a1/lam + a2/lam**2 + a3/lam**3

def nx(lam, T=25, source='Kato2002'):
    """KTP refractive index in x axis"""
    return sellmeier(lam, 3.29100, 0.04140, 0.03978, 9.35522, 31.45571)

def ny(lam, T=25, source='Konig2004'):
    """KTP refractive index in y axis"""
    if source == 'Kato2002':
        ny_20 = sellmeier(lam, 3.45018, 0.04341, 0.04597, 16.98825, 39.43799)
    elif source == 'Fan1987':
        b0, b1, b2, b3 = (2.19229, 0.83547, 0.04970, 0.01621)
        ny_25 = sqrt(b0 + b1 / (1 - b2 / lam**2) - b3 * lam**2)
    elif source == 'Konig2004':
        b0, b1, b2, b3 = (2.09930, 0.922683, 0.0467695, 0.0138408)
        ny_25 = sqrt(b0 + b1 / (1 - b2 / lam**2) - b3 * lam**2)
        
    ny1 = deltan_coeff(lam, 6.2897e-6, 6.3061e-6, -6.0629e-6, 2.6486e-6)
    ny2 = deltan_coeff(lam, -.14445e-8, 2.2244e-8, -3.5770e-8, 1.3470e-8)
    
    if source == 'Kato2002':
        return ny_20 + ny1 * (T - 20) + ny2 * (T - 20)**2
    else:
        return ny_25 + ny1 * (T - 25) + ny2 * (T - 25)**2

def nz(lam, T=25, source='Fradkin1999'):
    """KTP refractive index in z axis"""
    if source == 'Kato2002':
        nz_20 = sellmeier(lam, 4.59423, 0.06206, 0.04763, 110.80672, 86.12171)
    elif source == 'Fradkin1999':
        b0, b1, b2, b3, b4, b5 = (2.12725, 1.18431, .0514852, 
                                  0.66030, 100.00507, 9.68956e-3)
        nz_25 = sqrt(b0 + b1 / (1 - b2 / lam**2) + 
                     b3 / (1 - b4 / lam**2) - b5 * lam**2)
    elif source == 'Fan1987':
        b0, b1, b2, b3 = (2.25411, 1.06543, 0.05486, 0.02140)
        nz_25 = sqrt(b0 + b1 / (1 - b2 / lam**2) - b3 * lam**2)
        
    nz1 = deltan_coeff(lam, 9.9587e-6, 9.9228e-6, -8.9603e-6, 4.1010e-6)
    nz2 = deltan_coeff(lam, -1.1882e-8, 10.459e-8, -9.8136e-8, 3.1481e-8)
    
    if source == 'Kato2002':
        return nz_20 + nz1 * (T - 20) + nz2 * (T - 20)**2
    else:
        return nz_25 + nz1 * (T - 25) + nz2 * (T - 25)**2
    

    

class Cavity_Builder():
    def __init__(self, in_type, in_form, lam=1550e-6):
        self.type = in_type
        self.form = in_form
        self.lam = lam
        self.geometry = False
        self.optics = False
          
    def use(self):
        if self.type == 'nl':
            print('OPO: Optical parametric oscillator \nSHG: Second harmonic generation')
        elif self.type == 'filter':
            print('Filter: Frequency doubling \nMMC: Mode cleaning cavity \nMCC: Mode cleaning cavity')
            
    def info(self):
        if self.type == 'nl':
            if self.form == 'linear':
                print('Geometry parameters are:' + 
                      '\n (0) crystal length: l_cry [mm]' + 
                      '\n (1) crystal refractive index: n' +
                      '\n (2) spacing between curved mirrors: d [mm]' +
                      '\n (3) mirror curvature: R [mm]')

                print('\nOptics parameters are:' + 
                      '\n (0) output coupler reflection: 1-T_out' + 
                      '\n (1) input coupler reflection: 1-T_in' +
                      '\n (2) Additional internal cavity loss: L')
                
                print('\n      T_in       T_out     '+
                      '\n   -----(--[###]--)----->  '+ 
                      '\n'+
                      '\n        <- - - - -> d      ')
                
            elif self.form == 'triangle':
                print('Geometry parameters are:' + 
                      '\n (0) crystal length [mm]' + 
                      '\n (1) crystal refractive index' +
                      '\n (2) triangle width: d [mm]' +
                      '\n (3) triangle height: h [mm]' +
                      '\n (4) mirror curvature: R [mm]' +
                      '\n (5) astigmatism included [none, v, h]')
                
                print('\nOptics parameters are:' + 
                      '\n (0) output coupler reflection: 1-T_out' + 
                      '\n (1) input coupler reflection: 1-T_in' +
                      '\n (2) M3 reflection' +
                      '\n (4) Additional internal cavity loss: L')
                
                print('\n               ___ M3               ___ '+
                      '\n               / \                   |  '+
                      '\n             /     \                 |  '+
                      '\n           /         \               | h'+
                      '\n  T_in,R /             \ T_out,R     |  '+
                      '\n -------\-----[###]-----/-------->  _|_ '+
                      '\n'+
                      '\n        <- - - - - - - -> w             ')
                
            elif self.form == 'bowtie':
                print('Geometry parameters are:' + 
                      '\n (0) crystal length [mm]' + 
                      '\n (1) crystal refractive index' +
                      '\n (2) spacing between curved mirrors: d1 [mm]' +
                      '\n (3) spacing between flat mirrors: d2 [mm]' +
                      '\n (4) bowtie height: h [mm]'
                      '\n (5) mirror curvature: R [mm]'
                      '\n (6) astigmatism included [none, v, h]')
                
                print('\nOptics parameters are:' + 
                      '\n (0) output coupler reflection: 1-T_out' + 
                      '\n (1) input coupler reflection: 1-T_in' +
                      '\n (2) M3 reflection' +
                      '\n (3) M4 reflection' +
                      '\n (4) Additional internal cavity loss: L')
                
                print('\n         <- - - - -> d1       '+
                      '\n'+
                      '\n    M3,R /--[###]--\ M4,R     ___  '+ 
                      '\n          \       /            |   '+
                      '\n            \   /              |   '+
                      '\n              X                |   '+
                      '\n             / \               | h '+
                      '\n           /     \             |   '+
                      '\n    T_in /         \ T_out     |   '+
                      '\n   ----\-------------/----->  _|_  '+
                      '\n'+
                      '\n       <- - - - - - -> d2          ')
              
        elif self.type == 'filter': 
            if self.form == 'linear':
                print('Geometry parameters are:' + 
                      '\n (0) cavity length [mm]' +
                      '\n (1) mirror curvature [mm]')

                print('\nOptics parameters are:' + 
                      '\n (0) output coupler reflection' +
                      '\n (1) input coupler reflection' + 
                      '\n (2) Additional internal cavity loss')
                
                print('\n      T_in       T_out   '+
                      '\n   -----(---------)----> '+ 
                      '\n'+
                      '\n        <- - - - -> d      ')
            
            elif self.form == 'triangle':
                print('Geometry parameters are:' + 
                      '\n (0) triangle width: w [mm]' +
                      '\n (1) triangle height: h [mm]' +
                      '\n (2) mirror curvature: R [mm]' +
                      '\n (3) astigmatism included [none, v, h]')

                print('\nOptics parameters are:' + 
                      '\n (0) output coupler reflection: 1-T_out' +
                      '\n (1) input coupler reflection: 1-T_in' + 
                      '\n (2) M3 reflection' +
                      '\n (3) Additional internal cavity loss')
                
                print('\n              __ M3,R         ___ '+
                      '\n              /\               |  '+
                      '\n             /  \              |  '+
                      '\n            /    \             |h '+
                      '\n           /      \            |  '+
                      '\n     T_in /        \ T_out     |  '+
                      '\n    -----\----------/----->   _|_ '+
                      '\n'+
                      '\n         <- - - - - > w         ')
                
            elif self.form == 'bowtie':
                print('Geometry parameters are:' + 
                      '\n (0) spacing between curved mirrors [mm]' +
                      '\n (1) spacing between flat mirrors [mm]' +
                      '\n (2) bowtie hight [mm]'
                      '\n (3) mirror curvature [mm]'
                      '\n (4) astigmatism included [none, v, h]')
                
                print('\nOptics parameters are:' + 
                      '\n (0) output coupler reflection' + 
                      '\n (1) input coupler reflection' +
                      '\n (2) M3 reflection' +
                      '\n (3) M4 reflection' +
                      '\n (4) Additional internal cavity loss')
              
    def set_geometry(self, pam=[]):
        self.geometry = pam
        
        if self.type == 'nl':
            if self.form == 'linear':
                self.M = (abcd.prop(pam[0]/2)*abcd.infa(1,pam[1])*abcd.prop((pam[2]-pam[0])/2)*
                          abcd.curv(pam[3])*
                          abcd.prop((pam[2]-pam[0])/2)*abcd.infa(pam[1],1)*abcd.prop(pam[0]/2))
                
                self.lcav = pam[0]*pam[1] + (pam[2]-pam[0])
                self.tcav = self.lcav / (299792458*1e3)
                self.w0 = abcd.w2w(self.M, n1=pam[1], n2=pam[1], lam=self.lam)
                self.geometry_speclist = np.array([self.lcav, self.tcav, self.w0[0]])
                
                    
            elif self.form == 'triangle':
                _diag = np.sqrt((pam[2]/2)**2 + pam[3]**2)
                _theta = np.arcsin(pam[3] / _diag)
                _l3 = (pam[2]-pam[0])/2
                     
                self.M = (abcd.prop(pam[0]/2)*abcd.infa(1,pam[1])*abcd.prop(_l3)*
                          abcd.curv(pam[4],_theta/2,pam[5])*abcd.prop(2*_diag)*abcd.curv(pam[4],_theta/2,pam[5])*
                          abcd.prop(_l3)*abcd.infa(pam[1],1)*abcd.prop(pam[0]/2))
                
                self.lcav = pam[0]*pam[1] + 2*(_l3+_diag)
                self.tcav = self.lcav / (299792458*1e3)
                self.foldingangle = _theta
                self.w0 = abcd.w2w(self.M, n1=pam[1], n2=pam[1], lam=self.lam)
                self.geometry_speclist = np.array([self.lcav, self.tcav, self.foldingangle, self.w0[0]])
                
            elif self.form == 'bowtie':
                    _diag = np.sqrt(pam[4]**2 + (pam[2]+pam[3])**2/4)
                    _theta = np.arcsin(pam[4] / _diag)
                    _l3 = (pam[2]-pam[0])/2
                    
                    self.M = (abcd.prop(_diag+pam[3]/2)*abcd.curv(pam[5], _theta/2, pam[6])*
                             abcd.prop(_l3)*abcd.infa(pam[1],1)*abcd.prop(pam[0]/2))
                    
                    self.lcav = pam[0]*pam[1] + 2*_l3 + 2*_diag + pam[3]
                    self.tcav = self.lcav / (299792458*1e3)
                    self.foldingangle = _theta
                    self.w0 = abcd.w2w(self.M, n1=pam[1], n2=1, lam=self.lam)
                    self.geometry_speclist = np.array([self.lcav, self.tcav, self.foldingangle, self.w0[0], self.w0[1]])
                    
                    
        elif self.type == 'filter':
            if self.form == 'linear':
                self.M = abcd.prop(pam[0]/2)*abcd.curv(pam[1])*abcd.prop(pam[0]/2) 
                
                self.lcav = 2*pam[0]
                self.tcav = self.lcav / (299792458*1e3)
                self.w0 = abcd.w2w(self.M, n1=1, n2=1, lam=self.lam)
                self.geometry_speclist = np.array([self.lcav, self.tcav, self.w0[0]])
                
            elif self.form == 'triangle':
                _diag = np.sqrt((pam[0]/2)**2 + pam[1]**2)
                _theta = np.arccos(pam[1] / _diag)
                
                self.M = abcd.prop(pam[0]/2 + _diag) * abcd.curv(pam[2],_theta/2,pam[3]) * abcd.prop(pam[0]/2 + _diag)
                
                self.lcav = pam[0] + 2*_diag
                self.tcav = self.lcav / (299792458*1e3)
                self.foldingangle = _theta
                self.w0 = abcd.w2w(self.M, n1=1, n2=1, lam=self.lam)
                self.geometry_speclist = np.array([self.lcav, self.tcav, self.foldingangle, self.w0[0]])
                
            elif self.form == 'bowtie':
                _diag = np.sqrt(pam[2]**2 + (pam[0]+pam[1])**2/4)
                _theta = np.arcsin(pam[2] / _diag)

                self.M = abcd.prop(_diag+pam[1]/2)*abcd.curv(pam[3], _theta/2, pam[4])*abcd.prop(pam[0]/2)

                self.lcav = pam[0] + pam[1] + 2*_diag
                self.tcav = self.lcav / (299792458*1e3)
                self.foldingangle = _theta
                self.w0 = abcd.w2w(self.M, n1=1, n2=1, lam=self.lam)
                self.geometry_speclist = np.array([self.lcav, self.tcav, self.foldingangle, self.w0[0], self.w0[1]])
                
                
    def set_optics(self,pam=[], lcav=[]):
        self.optics = pam
        if lcav:
            self.lcav = lcav
            self.tcav = self.lcav / (299792458*1e3)
        
        T_out = 1-pam[0]
        T_rest = pam[1:-1]
        eff = 1
        for x in T_rest:
            eff *= x
        self.intcavloss = 1-eff*(1-pam[-1])
            
        gam_loss = (1-np.sqrt(1-self.intcavloss))/self.tcav
        gam_T = (1-np.sqrt(1-T_out))/self.tcav
        self.bw = (gam_T + gam_loss) / (2*np.pi)
        self.fsr = 1/self.tcav
        
        r = np.sqrt(1-T_out)*np.sqrt(1-self.intcavloss)
        self.finesse = np.pi*np.sqrt(r)/(1-r)
        self.esceff = T_out/(T_out+self.intcavloss)
        
        self.optics_speclist = np.array([self.fsr, self.bw, self.finesse, self.intcavloss, self.esceff])
        
    def print_specs(self):
        if not self.geometry:
            print('Please define cavity geometry using: set_geometry()')
        elif self.w0[0] == None:
            print('!!! No resonant mode possible !!!')
        else:
            if self.form == 'linear':
                spec_names = ['optical cavity length', 'cavity roundtrip time', 'waist size']
                spec_vals = self.geometry_speclist * np.array([1, 1e9, 1e3])
                spec_unit = ['mm', 'ns', 'um']
            elif self.form == 'triangle':
                spec_names = ['optical cavity length', 'cavity roundtrip time', 'folding angle', 'waist size']
                spec_vals = self.geometry_speclist * np.array([1, 1e9, 180/np.pi, 1e3])
                spec_unit = ['mm', 'ns', 'deg', 'um']
            elif self.form == 'bowtie':
                spec_names = ['optical cavity length', 'cavity roundtrip time', 'folding angle', '1st waist size', '2nd waist size']
                spec_vals = self.geometry_speclist * np.array([1, 1e9, 180/np.pi, 1e3, 1e3])
                spec_unit = ['mm', 'ns', 'deg', 'um','um'] 
            
            print('Geometry specs:')
            for i in range(len(spec_vals)):
                print(f'{spec_names[i]:<25}{spec_vals[i]:10.2f} {spec_unit[i]}')
            if not self.optics:    
                print('\nPlease define cavity optics using: set_optics()')
            else:
                spec_names = ['free spectral range', 'bandwidth', 'finesse', 'intra cavity loss', 'escape efficiency']
                spec_vals = self.optics_speclist * np.array([1e-6, 1e-6, 1, 100, 100])
                spec_unit = ['MHz', 'MHz', '', '%', '%']

                print('\nOptics specs:')
                for i in range(len(spec_vals)):
                    print(f'{spec_names[i]:<25}{spec_vals[i]:10.2f} {spec_unit[i]}')                
            
            
    def find_geometry(self, pam, pam_i, pam_v):
        self.set_geometry(pam)
        geometry_out = np.zeros((len(self.geometry_speclist), len(pam_v)))
        
        for i,v in enumerate(pam_v):
            pam[pam_i] = v 
            self.set_geometry(pam)
            geometry_out[:,i] = self.geometry_speclist
            
        return geometry_out
    
class Cavity_Plotter():
    def __init__(self, cav):
        self.cav = cav
        
    def xy_rot(self,x,y,x0,y0,a):
        r = np.sqrt((x-x0)**2 + (y-y0)**2)
        phi = np.arctan2(y-y0,x-x0)
        x_out = r * np.cos(phi+a) + x0
        y_out = r * np.sin(phi+a) + y0
        return x_out, y_out

    def mirror(self,s, angle=0, diameter=0.5, form='flat'):
        x0, y0 = s
        inch = 25.4 

        xc = 1           # coating thickness
        xs = 6             # substrate thickess
        y = inch*diameter  # substrate diameter
        theta = angle*180/np.pi

        substrate_top = plt.Rectangle((x0, y0), xs, y/2, angle=angle, fc='C0', alpha=0.5)
        substrate_bot = plt.Rectangle((x0, y0), xs, -y/2, angle=angle, fc='C0', alpha=0.5)        
        coating_top = plt.Rectangle((x0, y0), xc, y/2, angle=angle, fc='k', alpha=1)
        coating_bot = plt.Rectangle((x0, y0), xc, -y/2, angle=angle, fc='k', alpha=1)

        return [substrate_top, substrate_bot, coating_top, coating_bot]


    def plot_mirror(self,s,angle,diameter,gca):
        M = self.mirror(s,angle, diameter)
        return gca.add_patch(M[0]), gca.add_patch(M[1]), gca.add_patch(M[2]) , gca.add_patch(M[3]) 
    
    def crystal(self, s, lcry, wcry):
        x0,y0 = s
        crystal = plt.Rectangle((x0-lcry/2, y0-wcry/2), lcry, wcry, fc='C2', alpha=0.5)
        return crystal
    
    def plot_crystal(self, s, lcry, wcry, gca):
        x0,y0 = s
        crystal = plt.Rectangle((x0-lcry/2, y0-wcry/2), lcry, wcry, fc='C2', alpha=0.5)
        return gca.add_patch(crystal)
    

    def beam(self,z, w0):
        zR = np.pi*w0**2/self.cav.lam
        w = w0*np.sqrt(1+(z/zR)**2)
        return w

    def plot_beam1(self,s0,s1, z0, w0, ax):
        x0,y0 = s0
        x1,y1 = s1

        x = np.linspace(x0,x1,100)
        y = np.linspace(y0,y1,100)
        z = np.sqrt(x*x + y*y)
        angle = np.arctan((y1-y0)/(x1-x0))

        w = self.beam(z0 + z, w0)

        xw_top = x - w * np.sin(angle)
        yw_top = y + w * np.cos(angle)
        xw_bot = x + w * np.sin(angle)
        yw_bot = y - w * np.cos(angle)

        #ax0 = ax.plot(x,y, 'k--', alpha=.5)
        #ax_top = ax.plot(xw_top,yw_top, 'k', alpha=0.5)
        #ax_bot = ax.plot(xw_bot,yw_bot, 'k', alpha=0.5)
        ax_filltopy = ax.fill_between(xw_top, yw_top, yw_bot, color='C3', alpha=1)
        ax_fillboty = ax.fill_between(xw_bot, yw_bot, yw_top, color='C3', alpha=1)
        ax_filltopx = ax.fill_betweenx(yw_top, xw_top, xw_bot, color='C3', alpha=1)
        ax_fillbotx = ax.fill_betweenx(yw_bot, xw_bot, xw_top, color='C3', alpha=1)

    def plot_beam2(self,s0,d,angle, z0, w0, ax):
        x0,y0 = s0
        x1 = x0 + d * np.cos(angle*np.pi/180)
        y1 = y0 + d * np.sin(angle*np.pi/180)

        x = np.linspace(x0,x1,100)
        y = np.linspace(y0,y1,100)
        z = np.linspace(z0,z0+d,100)

        w = beam(z, w0)
        xw_top = x - w * np.sin(angle*np.pi/180)
        yw_top = y + w * np.cos(angle*np.pi/180)
        xw_bot = x + w * np.sin(angle*np.pi/180)
        yw_bot = y - w * np.cos(angle*np.pi/180)

        #ax0 = ax.plot(x,y, 'k--', alpha=.5)
        #ax_top = ax.plot(xw_top,yw_top, 'k', alpha=0.5)
        #ax_bot = ax.plot(xw_bot,yw_bot, 'k', alpha=0.5)
        ax_filltopy = ax.fill_between(xw_top, yw_top, yw_bot, color='C3', alpha=1)
        ax_fillboty = ax.fill_between(xw_bot, yw_bot, yw_top, color='C3', alpha=1)
        ax_filltopx = ax.fill_betweenx(yw_top, xw_top, xw_bot, color='C3', alpha=1)
        ax_fillbotx = ax.fill_betweenx(yw_bot, xw_bot, xw_top, color='C3', alpha=1)


        return #ax_top, ax_bot, ax_filltopy, ax_fillboty, ax_filltopx, ax_fillbotx
    
    def plot_filter_triangle(self, lims=[]):
        w,h,_,_ = self.cav.geometry
        w0 = self.cav.w0[0]
        
        theta1 = np.arctan(2*h/w)/2
        theta2 = np.arctan(w/(2*h))
        x_anchors = np.array([0,w/2,-w/2,0])
        y_anchors = np.array([0,0,0,h])
        anchors = np.array([x_anchors,y_anchors]).T
        
        fig, ax = plt.subplots(dpi= 150)
        
        if w0:
            self.plot_beam1(anchors[0],anchors[1],0, w0, ax=ax)
            self.plot_beam1(anchors[0],anchors[2],0, w0, ax=ax)
            self.plot_beam1(anchors[1],anchors[3],0, w0, ax=ax)
            self.plot_beam1(anchors[2],anchors[3],0, w0, ax=ax)
        else:
            pass
        
        self.plot_mirror(anchors[1], -theta1*180/np.pi, 0.5, gca=plt.gca())
        self.plot_mirror(anchors[2], 180+theta1*180/np.pi, 0.5, gca=plt.gca())
        self.plot_mirror(anchors[3], 90, 0.5, gca=plt.gca())
        
        ax.axis('scaled')
        if lims:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
        else:
            ax.set_xlim(-(w/2+20), w/2+20)
            ax.set_ylim(-10,h+10)
        ax.grid(alpha=0.3)
        plt.show()
        
    def plot_nl_bowtie(self, lims=[], axis_scaling='scaled'):
        lcry,_,d1,d2,h,_,_ = self.cav.geometry
        w0 = self.cav.w0
        
        theta = 2*self.cav.foldingangle * 180/np.pi
        x_anchors = np.array([0,d2/2,2*d2,-d2/2,-2*d2,0,d1/2,-d1/2])
        y_anchors = np.array([0,0,0,0,0,h,h,h])
        anchors = np.array([x_anchors,y_anchors]).T
        
        fig, ax = plt.subplots(dpi= 200)
        
        if w0[0]:
            self.plot_beam1(anchors[0],anchors[2],0, w0[1], ax=ax)
            self.plot_beam1(anchors[1],anchors[-1],0, w0[1], ax=ax)
            self.plot_beam1(anchors[0],anchors[4],0, w0[1], ax=ax)
            self.plot_beam1(anchors[3],anchors[-2],0, w0[1], ax=ax)
            self.plot_beam1(anchors[5],anchors[-1],0, w0[0], ax=ax)
            self.plot_beam1(anchors[5],anchors[-2],0, w0[0], ax=ax)
        else:
            pass
        
        self.plot_crystal(anchors[5], lcry, 1, gca=plt.gca())
        self.plot_mirror(anchors[1], -theta, 0.25, gca=plt.gca())
        self.plot_mirror(anchors[3], 180+theta, 0.25, gca=plt.gca())
        self.plot_mirror(anchors[6], theta, 0.25, gca=plt.gca())
        self.plot_mirror(anchors[7], 180-theta, 0.25, gca=plt.gca())
        
        ax.axis(axis_scaling)
        if lims:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
        else:
            ax.set_xlim(-(d2/2+20), d2/2+20)
            ax.set_ylim(-10,h+10)
        ax.grid(alpha=0.3)
        plt.show()
        
        
    def plot_nl_linear(self, lims=[], axis_scaling='scaled'):
        lcry,_,d,_ = self.cav.geometry
        w0 = self.cav.w0[0]
        
        x_anchors = np.array([0,d/2,2*d,-d/2,-2*d])
        y_anchors = np.array([0,0,0,0,0])
        anchors = np.array([x_anchors,y_anchors]).T
        
        fig, ax = plt.subplots(dpi= 150)
        
        if w0:
            self.plot_beam1(anchors[0],anchors[2],0, w0, ax=ax)
            self.plot_beam1(anchors[0],anchors[4],0, w0, ax=ax)
        else:
            pass
        
        self.plot_crystal(anchors[0], lcry, 1, gca=plt.gca())
        self.plot_mirror(anchors[1], 0, 0.25, gca=plt.gca())
        self.plot_mirror(anchors[3], 180, 0.25, gca=plt.gca())
        
        ax.axis(axis_scaling)
        if lims:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
        else:
            ax.set_xlim(-(d/2+20), d/2+20)
        ax.grid(alpha=0.3)
        plt.show()
        
            
            
