*----------------------------------------------------------------------*
* COMPILE USING
*  f2py -m interpolate -c interpolate.f	

        subroutine interpolate(N, xn, bias_x, dx, fx,
     +	Nxg,  Fx_n)

Cf2py intent(in)  N   
Cf2py intent(in)  xn                                        
Cf2py intent(in)  bias_x
Cf2py intent(in)  dx
Cf2py intent(in)  fx
Cf2py intent(in)  mdl
Cf2py intent(in)  Nxg
Cf2py intent(out) Fx_n

        implicit none
	real*8   PI
	integer  N
	real*8   xn(N)
	real*8   bias_x, dx
	integer  Nxg
	real*8   fx(Nxg)
	integer  p
	real*8   fi, hx
	integer  i
        real*8   Fx_n(N)
        
        
        do p=1,N
        fi = 1+(xn(p)-bias_x)/dx;             !i index of particle's cell 
     	i  = int(fi);
     	hx = fi-dble(i);                      !fractional x position in cell
  	
     	Fx_n(p)=fx((i))*(1-hx)+ fx((i+1))*hx; 
     	
     	end do
     	
     	end subroutine

