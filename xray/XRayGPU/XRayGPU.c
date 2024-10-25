// simple random number generator based on:
// http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
// (may not have ideal characteristics, but it is fast!)

unsigned rand_xorshift(unsigned *rng_state)
{
  // Xorshift algorithm from George Marsaglia's paper
  *rng_state ^= ((*rng_state) << 13);
  *rng_state ^= ((*rng_state) >> 17);
  *rng_state ^= ((*rng_state) << 5);
  return *rng_state;
}


unsigned wang_hash(unsigned seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}


// basic vector operations
static inline void subvecvec3(float *vec1,float *vec2,float *out)
{
  int outidx;

  for (outidx=0;outidx < 3; outidx++) {
    out[outidx] = vec1[outidx] - vec2[outidx];
    
  }
}

static inline float dotvecvec3(float *vec1,float *vec2)
{
  int sumidx;
  float val=0.0f;
  for (sumidx=0;sumidx < 3; sumidx++) {
    val = val + vec1[sumidx]*vec2[sumidx];
    
  }
  return val;
}

static inline void projectvecvec3(float *vec1,float *vec2,float *output)
{
  // projection in the direction of vec1 of vec2
  float coeff;
  coeff = dotvecvec3(vec1,vec2)/dotvecvec3(vec1,vec1);

  output[0]=coeff*vec1[0];
  output[1]=coeff*vec1[1];
  output[2]=coeff*vec1[2];
}

static inline void crossvecvec3(float *vec1,float *vec2,float *output)
{
  /* 
     vec1 cross vec2 

   |   i     j     k    |
   | vec10 vec11  vec12 |
   | vec20 vec21  vec22 |
  */
  output[0] = vec1[1]*vec2[2]-vec1[2]*vec2[1];
  output[1] = vec1[2]*vec2[0]-vec1[0]*vec2[2];
  output[2] = vec1[0]*vec2[1]-vec1[1]*vec2[0];
}

static inline void multmatmat3(float *mat1,float *mat2, float *mat3)
{
  // multiply 3x3 matrices mat1*mat2 -> mat3
  // mat3_ij = mat1_ik*mat2_kj
  unsigned i,j,k;
  float accum;
  
  for (i=0; i < 3; i++) {
    for (j=0;j < 3; j++) {
      accum=0.0f;
      for (k=0;k < 3; k++) {
	accum+=mat1[3*i+k]*mat2[3*k+j];
      }
      mat3[3*i+j] = accum;
    }
  }
}

// X-Ray absorption code
__constant const float electronrestenergyeV=510998.9100f;


float K_N_Integral_CDF_Indefinite_Unnormalized(float theta, float alpha0)
{
  // The scattering cross section per unit solid angle
  // is given by the Klein-Nishina formula: 
  float K_N_integral;
  
  // dsigmadOmega=(1./(1+alpha0.*(1-cos(theta)))).^2 .* ((1+cos(theta).^2)/2.0) .* (1 + alpha0.^2 .* (1-cos(theta)).^2./((1+cos(theta).^2) .* (1+alpha0.*(1-cos(theta)))));
  // alternate form, replacing theta with x, multiplying in solid angle unit sin(theta)*dtheta*dphi and integrating over phi=0...2pi
  // integral of (1/(1+F*(1-cos(x))))^2 * ( (1/(1+F*(1-cos(x)))) + 1 + F*(1-cos(x)) - (sin(x))^2)*(2*pi*sin(x)/2) *  dx
  // where F=alpha0
  // by Wolfram alpha:
  // pi*(-2*F*cos(theta)+ (-2-6*F-5*F^2+2*F*(1+2*F)*cos(theta))/((1+F-F*cos(theta))^2) + 2*(-2-2*F+F^2)*log(1+F-F*cos(theta)))/(2*F^3)
  K_N_integral = M_PI_F*(-2.0f*alpha0*cos(theta) + (-2.0f-6.0f*alpha0-5.0f*pow(alpha0,2.0f)+2.0f*alpha0*(1.0f+2.0f*alpha0)*cos(theta))/pow(1.0f+alpha0-alpha0*cos(theta),2.0f) + 2.0f*(-2.0f-2.0f*alpha0+pow(alpha0,2.0f))*log(1.0f+alpha0-alpha0*cos(theta)))/(2.0f*pow(alpha0,3.0f));

  
  return K_N_integral; // to be an actual probability, values at two thetas need to be subtracted, then normalized by K_N_Integral_EntireSphere

}

float K_N_Integral_CDF(float theta,float alpha0)
// integral from 0 up to theta, needs to be normalized by K_N_Integral_EntireSphere
{
  return K_N_Integral_CDF_Indefinite_Unnormalized(theta,alpha0)-K_N_Integral_CDF_Indefinite_Unnormalized(0.0f,alpha0);
  
}

float K_N_Integral_EntireSphere(float alpha0)
{
  //return M_PI_F*(-2.0f*alpha0*cos(M_PI_F) + (-2.0f-6.0f*alpha0-5.0f*pow(alpha0,2.0f)+2.0f*alpha0*(1.0f+2.0f*alpha0)*cos(M_PI_F))/pow(1.0f+alpha0-alpha0*cos(M_PI_F),2.0f) + 2.0f*(-2.0f-2.0f*alpha0+pow(alpha0,2.0f))*log(1.0f+alpha0-alpha0*cos(M_PI_F)))/(2.0f*pow(alpha0,3.0f));
  return K_N_Integral_CDF(M_PI_F,alpha0);
  
}

float K_N_PDF(float theta,float alpha0)
{
  // derivative of the above CDF... a PDF as a function of theta.
  // Needs to be normalized by K_N_Integral_EntireSphere()

  float pdfval;
  pdfval = pow(1.0f/(1.0f+alpha0*(1.0f-cos(theta))),2.0f) * ( (1.0f/(1.0f+alpha0*(1.0f-cos(theta)))) + 1.0f + alpha0*(1.0f-cos(theta)) - pow(sin(theta),2.0f)) * (2.0f*M_PI_F*sin(theta)/2.0f);

  if (pdfval < 1e-8f) {
    // if pdf is too small (it is the derivative of something
    // that integrates over pi to O(1), when we attempt Newton's
    // method we will be dividing by zero giving large errors.
    // so we bound how small a result we will give
    pdfval=1e-8f;
  }
  return pdfval;
}

float K_N_Integral_CDF_Inverse(float probability,float alpha0)
{
  // Given a probability between 0 and 1, plug that into the
  // inverse scattering CDF of the Klein-Nishina formula
  // to solve for and return the angle theta of scattering
  
  // use Newton's method solver to evaluate K_N_Integral_CDF(theta,alpha0)/K_N_Integral_EntireSphere - probability = 0.0
  // mutiply through by K_N_Integral_EntireSphere:
  //  K_N_Integral_CDF(theta,alpha0) - probability*K_N_Integral_EntireSphere = 0.0
  float newtheta=M_PI_F/2.0f;
  float theta;
  float normalization;
  float denormalized_probability;
  unsigned cnt=0;
  
  normalization = K_N_Integral_EntireSphere(alpha0);

  denormalized_probability=probability * normalization;
  //printf("alpha0=%f\n",alpha0);
  //printf("normalization=%f; denormalized_probability=%f\n",normalization,denormalized_probability);
  
  //printf("Start Newton;probability=%f\n",probability);
  do {
    theta = newtheta;
    //printf("newtheta=%f\n",newtheta);
    newtheta = theta - ( K_N_Integral_CDF(theta,alpha0) - denormalized_probability) / K_N_PDF(theta,alpha0);

    cnt++;
  } while (fabs(theta-newtheta) > 1e-3f && cnt < 10);  // experimentally we usually get convergence within 5 iterations. But for extreme probabilities (close to 0 and 1, mapping to 0 and pi, respectively) we sometimes don't get convergence. Presumably some of this is due to numerical roundoff error from single precision. So we just stop iterating after 10 counts.  
  //printf("End Newton; newtheta=%f\n",newtheta);

  if (isnan(newtheta)) {
    printf("K_N_Integral_CDFInverse returns NaN\n");
  }
  
  return newtheta;
}


__kernel void absorp(
		     float x_bnd0, float dx, unsigned nxbnd,
		     float y_bnd0, float dy, unsigned nybnd,
		     float z_bnd0, float dz, unsigned nzbnd,
		     float detector_z,
		     float logE0, float dlogE, unsigned nlogE, // Note that these are the log base 10 initial and step energies for the material data
		     __global const float *matl_data,__global const float *matl_rho,
		     unsigned n_matl,
		     __global const unsigned *matl_volumetric,
		     unsigned photonpos_nidx1,
		     __global const float *photonpos, /* compute_dimensions[0] x compute_dimensions[1] * 3 where compute_dimensions[1] is known from photonpos_nidx1 */
		     __global const float *photonvec, /* compute_dimensions[0] x compute_dimensions[1] * 3 where compute_dimensions[1] is known from photonpos_nidx1 ...  NOTE: Each vector is assumed to be normalized ***!!! */
		     __global const float *photon_logE,
		     unsigned num_photons,
		     __global int *detector_photons,
		     unsigned seed)
{
  unsigned photon_idx0 = get_global_id(0);
  unsigned photon_idx1 = get_global_id(1);
  float eps=dx/100.0f; // small increment
  unsigned iterlimit,itercnt;
  unsigned photoncnt;
  unsigned need_new_photon;
  float log_energy=0.0f;
  float photonmtx[9]={0.0f};
  bool photon_gone=false;
  
  float pos_x,pos_y,pos_z;
  float vec_x,vec_y,vec_z;

  float gridpos_x,gridpos_y,gridpos_z;
  
  float newpos_x=0.0f,newpos_y=0.0f,newpos_z=0.0f;
  float newvec_x=0.0f,newvec_y=0.0f,newvec_z=0.0f;
  
  union {
    unsigned InfUnsigned; // 0x7f800000
    float InfFloat;
  } Infval;
  
  float inf;

  unsigned rng_seed;

  // Create variable with 'infinity'
  Infval.InfUnsigned = 0x7f800000;
  inf = Infval.InfFloat; // Aliasing int->float through a union is OK because OpenCL spec specifically allows this  (weaker strict aliasing rules than traditional C)

  // Limit the maximum number of iterations through
  // which we will track a particle by the largest
  // of the sizes of all three axes, times 10. 
  iterlimit=nxbnd;
  if (iterlimit < nybnd) {
    iterlimit=nybnd;
  }
  if (iterlimit < nzbnd) {
    iterlimit=nzbnd;
  }
  iterlimit*=10;

  // Use a hash function to improve the random number seed. Seed is based on which photon position we are computing
  rng_seed = wang_hash((get_global_id(0) + (get_global_id(1) << 16)) ^ seed);



  
  need_new_photon=1;

  // iterate over each step of each of the photons we are supposed to compute per position in this kernel run

  // need_new_photon will be set once something happens to the photon we are working on: 
  // (absorbed, goes out of domain, hits detector)

  for (photoncnt=0;photoncnt < num_photons;photoncnt += need_new_photon) {
    //printf("photoncnt=%u\n",photoncnt);

    if (need_new_photon) {
      // Initialize photon position to the starting position we were given
      newpos_x = photonpos[photonpos_nidx1*photon_idx0*3 + photon_idx1*3 + 0];
      newpos_y = photonpos[photonpos_nidx1*photon_idx0*3 + photon_idx1*3 + 1];
      newpos_z = photonpos[photonpos_nidx1*photon_idx0*3 + photon_idx1*3 + 2];
      
      // Initialize direction unit vector to the starting direction we were given
      newvec_x = photonvec[photonpos_nidx1*photon_idx0*3 + photon_idx1*3 + 0];
      newvec_y = photonvec[photonpos_nidx1*photon_idx0*3 + photon_idx1*3 + 1];
      newvec_z = photonvec[photonpos_nidx1*photon_idx0*3 + photon_idx1*3 + 2];
    
      // construct transformation matrix from lab coordinates to
      // photon vector coordinates
      //
      // 3rd row of matrix is vector photon vector
      
      
      photonmtx[2*3 + 0] = newvec_x;
      photonmtx[2*3 + 1] = newvec_y;
      photonmtx[2*3 + 2] = newvec_z;
      
      // apply Gram-schmidt  orthonormalization
      // to determine 2nd row
      // use the more-orthogonal to newvec of x or y
      float v2vec[3],projv2[3];
      if (newvec_x > newvec_y) {
	v2vec[0]=0.0f;
	v2vec[1]=1.0f;
	v2vec[2]=0.0f;
      } else {
	v2vec[0]=1.0f;
	v2vec[1]=0.0f;
	v2vec[2]=0.0f;
      }
      projectvecvec3(&photonmtx[2*3+0],v2vec,projv2); // calculate the projection in the newvec direction of v2
      subvecvec3(v2vec,projv2,&photonmtx[1*3+0]); // 1st orthogonal vector
      // 1st row should be the cross product of 2nd and 3rd rows
      crossvecvec3(&photonmtx[1*3+0],&photonmtx[2*3+0],&photonmtx[0*3+0]);
      
      // photonmtx * vector in lab space -> coordinates with 3rd component parallel to photon trajectory
    

      // extract energy from the array were were given and make sure it is reasonable. 
      log_energy = photon_logE[photonpos_nidx1*photon_idx0 + photon_idx1];

      if (log_energy > 10.0f || log_energy < 2.0f || isnan(log_energy) || isinf(log_energy)) {
	//printf("Unreasonable log_energy input idx0=%u\n",photon_idx0);
	printf("Unreasonable log_energy input idx0=%u, idx1=%u, log_energy=%f\n",photon_idx0,photon_idx1,log_energy);
      }
    
      photon_gone=false;
      need_new_photon=0;
      itercnt=0; // numver of iterations of this particular photon
    }


    float nextisect_x,nextisect_y,nextisect_z;
    float nextisect_xt,nextisect_yt,nextisect_zt;
    float nextisect_t;
    
    
    float newpos_xbndidx,newpos_ybndidx,newpos_zbndidx;
    float midposx,midposy,midposz;
    float matl_idx_x,matl_idx_y,matl_idx_z;
    
    
    // Update position and direction
    pos_x=newpos_x;
    pos_y=newpos_y;
    pos_z=newpos_z;
    vec_x=newvec_x;
    vec_y=newvec_y;
    vec_z=newvec_z;
    
    
    // Positions relative to the element boundaries
    gridpos_x = (pos_x-x_bnd0)/dx;
    gridpos_y = (pos_y-y_bnd0)/dy;
    gridpos_z = (pos_z-z_bnd0)/dz;
    
    //printf("pos_x=%f, pos_y=%f, pos_z=%f\n",pos_x,pos_y,pos_z);
    
    
    // identify next intersections with boundary planes in
    // x, y, and z
    
    // x boundary plane
    if (vec_x > 0.0f) {
      nextisect_x = ceil(gridpos_x+eps);      
    } else if (vec_x < 0.0f) {
      nextisect_x = floor(gridpos_x-eps);
    } else {
      nextisect_x = inf;      
    }
    // newpos*dx = pos*dx+vec*t
    // newpos_x*dx = pos_x*dx + vec_x*t
    // t = (newpos_x-pos_x)*dx/vec_x
    nextisect_xt = (nextisect_x-gridpos_x)*dx/vec_x;
    
    // y boundary plane
    if (vec_y > 0.0f) {
      nextisect_y = ceil(gridpos_y+eps);      
    } else if (vec_y < 0.0f) {
      nextisect_y = floor(gridpos_y-eps);
    } else {
      nextisect_y = inf;
    }
    nextisect_yt = (nextisect_y-gridpos_y)*dy/vec_y;
    
    // z boundary plane
    if (vec_z > 0.0f) {
      nextisect_z = ceil(gridpos_z+eps);      
    } else if (vec_z < 0.0f) {
      nextisect_z = floor(gridpos_z-eps);
    } else {
      nextisect_z = inf;
    }
    nextisect_zt = (nextisect_z-gridpos_z)*dz/vec_z;
    
    nextisect_t = inf;
    
    // select nearest boundary plane
    if (isfinite(nextisect_xt) && nextisect_xt <= nextisect_yt && nextisect_xt <= nextisect_zt) {
      nextisect_t = nextisect_xt;
    }
    if (isfinite(nextisect_yt) && nextisect_yt <= nextisect_xt && nextisect_yt <= nextisect_zt) {
      nextisect_t = nextisect_yt;
    }
    if (isfinite(nextisect_zt) && nextisect_zt <= nextisect_xt && nextisect_zt <= nextisect_yt) {
      nextisect_t = nextisect_zt;
    }
    
    // nextisect_t is distance along the propagation line
    // to the nearest boundary plane interface. It has
    // units of meters. 
    
    // Calculate new position (at next intersection)
    newpos_x = pos_x + nextisect_t * vec_x;
    newpos_y = pos_y + nextisect_t * vec_y;
    newpos_z = pos_z + nextisect_t * vec_z;
    
    // Calculate positions relative to element boundaries
    // (will be used below to see if we have exited domain)
    newpos_xbndidx = (newpos_x-x_bnd0)/dx;
    newpos_ybndidx = (newpos_y-y_bnd0)/dy;
    newpos_zbndidx = (newpos_z-z_bnd0)/dz;
    
    // Calculate midpoint of the propagation
    // segment we are operating on 
    midposx = (pos_x+newpos_x)/2.0f;
    midposy = (pos_y+newpos_y)/2.0f;
    midposz = (pos_z+newpos_z)/2.0f;
    
    // Find the index into the volumetric material ID
    // matrix corresponding to the segment midpoint.
    // We do this by comparing with the left boundary
    // of the matrix element and rounding down (floor)
    matl_idx_x = floor((midposx-x_bnd0)/dx);
    matl_idx_y = floor((midposy-y_bnd0)/dy);
    matl_idx_z = floor((midposz-z_bnd0)/dz);

    // If we are inside the material matrix...
    if (matl_idx_x >= 0.0f && matl_idx_x < ((float)(nxbnd-1)) &&
	matl_idx_y >= 0.0f && matl_idx_y < ((float)(nybnd-1)) &&
	matl_idx_z >= 0.0f && matl_idx_z < ((float)(nzbnd-1))) {
      unsigned material;
      float logE_idx;
      float compton;
      float photoe;
      float rho;
      float compton_atten;
      float draw;
      float P_compton;
      
	// Look up the material index
      material = matl_volumetric[(nybnd-1)*(nzbnd-1)*((unsigned)matl_idx_x) + (nzbnd-1)*((unsigned)matl_idx_y) + ((unsigned)matl_idx_z)];
      
      //printf("nybnd=%d, nzbnd=%d,matl_idx_x=%f,matl_idxy=%f,matl_idxz=%f matl_volumetric[0]=%u\n",nybnd,nzbnd,matl_idx_x,matl_idx_y,matl_idx_z,matl_volumetric[0]);
      
      // Figure out the index of the logE entry
      // corresponding to our particular log energy
      logE_idx = round((log_energy-logE0)/dlogE);
      
      if (logE_idx < 0.0f) logE_idx=0.0f;
      if (logE_idx >= nlogE) logE_idx = nlogE-1;
      
      // if we are photoelectrically absorbed, then photon_gone=true
      // if we are Compton scattered then assign newvec and photonmtx according to the new direction, and also reduce the particle energy
      
      // extract Compton and photoelectric
      // absorption coefficients in m^2/kg
      //printf("nlogE=%u, material=%u,logE_idx=%f\n",nlogE,material,logE_idx);
      
      // matl_data is n_materials by 2 by nlogE 
      compton = matl_data[(nlogE*2)*material + nlogE*0 + ((unsigned)logE_idx)];
      photoe = matl_data[(nlogE*2)*material + nlogE*1 + ((unsigned)logE_idx)];
      
      rho = matl_rho[material];
      
      //printf("compton=%f, photoe=%f, rho=%f\n",compton,photoe,rho);
      
      compton_atten=rho*compton; // density * coefficient... This would be the exponent in the exp(-rho*c*x) where in this case x will be  represented by the distance parameter t.
      
      // absorption is exponentially distributed; the rate parameter lambda is
      // rho*compton or rho*photoe
      // ... the CDF of the exponential distribution
      // is 1-exp(-lambda*x) for x >= 0
      // which would be the probability of absorption before x
      // P(compton) = 1-exp(-compton_atten * nextisect_t)
      // So we take a uniform random draw. If the value is less than
      // P(compton) we say it was absorbed. 
      P_compton = (1.0f-exp(-compton_atten * nextisect_t));
      
      draw = rand_xorshift(&rng_seed)*1.0f/4294967295.0f; // random draw between 0 and 1
      //printf("draw=%f; P_compton=%f\n",(double)draw,(double)P_compton);
      if (draw < P_compton) {
	// Compton scattered
	  // into what direction?
	
	float alpha0,theta,energy_out;
	float phi;
	
	float rotmtx1[9];
	float rotmtx2[9];
	float tempmtx[9];
	
	// alpha0 is the ratio of the photon energy
	// to the rest energy of an electron
	alpha0=pow(10.0f,log_energy)/electronrestenergyeV;
	
	
	draw = rand_xorshift(&rng_seed)*1.0f/4294967295.0f; // another random draw between 0 and 1
	theta = K_N_Integral_CDF_Inverse(draw,alpha0);  // based on the draw and the inverse CDF of the Klein-Nishina angular probability density, evaluate the scattering angle
	

	// Another random draw to determine phi angle between 0 and 2pi
	phi = rand_xorshift(&rng_seed)*2.0f*M_PI_F/4294967295.0f;  // 429... is (2^32-1), maximum value of an opencl unsigned int
	
	// in coordinates where z was the original direction,
	// rotation matrices:
	// [ outgoing sideways ]   [  cos(theta) 0  -sin(theta) ] [ cos(phi)  -sin(phi)  0 ][ orig sideways ]
	// [ outgoing up       ] = [      0      1     0        ] [ sin(phi)   cos(phi)  0 ][ orig up       ]
	// [ outgoing parallel ]   [ sin(theta)  0  cos(theta)  ] [    0           0     1 ][ orig parallel ]
	
	// multiply these matrices on the right by a vector in incoming propagation
	// coordinates to get a vector in scattered coordinates. 
	//
	// If the incoming vector is:
	//   [ orig sideways ]   [ Q11 Q12 Q13 ] [ orig lab x ]
	//   [ orig up       ] = [ Q21 Q22 Q23 ] [ orig lab y ]
	//   [ orig parallel ]   [ Q31 Q32 Q33 ] [ orig lab z ]
	//                                Q = photon_mtx
	// [ Q31 ; Q32 ; Q33 ] = orig propagation unit vector
	//  * Other rows of Q constructed by Gram-Schmidt orthonormalization
	// Then outgoing_vector = bottom row of ([ above Givens rotation matrices ] * Q )
	//  outgoing_Q  = [ above Givens rotation matrices ] * Q
	rotmtx1[0]=cos(theta) ; rotmtx1[1]=0.0f   ; rotmtx1[2]=-sin(theta);
	rotmtx1[3]=0.0f       ; rotmtx1[4]=1.0f   ; rotmtx1[5]=0.0f;
	rotmtx1[6]=sin(theta) ; rotmtx1[7]=0.0f   ; rotmtx1[8]=cos(theta);
	
	rotmtx2[0]=cos(phi) ; rotmtx2[1]=-sin(phi) ; rotmtx2[2]=0.0f;
	rotmtx2[3]=sin(phi) ; rotmtx2[4]=cos(phi)  ; rotmtx2[5]=0.0f;
	rotmtx2[6]=0.0f     ; rotmtx2[7]=0.0f      ; rotmtx2[8]=1.0f;
	
	multmatmat3(rotmtx2,photonmtx,tempmtx);
	multmatmat3(rotmtx1,tempmtx,photonmtx);
	
	// newvec is last row of photonmtx
	newvec_x=photonmtx[2*3+0];
	newvec_y=photonmtx[2*3+1];
	newvec_z=photonmtx[2*3+2];
	
	// Calculate energy of scattered photon
	energy_out = alpha0*electronrestenergyeV/(1.0f+alpha0*(1.0f-cos(theta)));
	log_energy=log10(energy_out);
	
	
      } else {
	  // check for photoelectric absorption
	float P_photoe;
	
	P_photoe = (1.0f-exp(-rho*photoe * nextisect_t));
	draw = rand_xorshift(&rng_seed)*1.0f/4294967295.0f; // random draw between 0 and 1
	
	//printf("draw=%f; P_photoe=%f\n",draw,P_photoe);
	
	if (draw < P_photoe) {
	  //printf("Photon photoelectrically absorbed\n");
	  photon_gone=true; 
	}
	
      }
      
    }
    
    //printf("newpos_x=%f, newpos_y=%f, newpos_z=%f\n",newpos_x,newpos_y,newpos_z);
    
    if (newpos_z >= detector_z || newpos_zbndidx >= ((float)nzbnd)) {
      // photon has past or is headed toward detector
      
      // Bump it the rest of the way (or back if need be)
      nextisect_t = (detector_z-newpos_z)/vec_z;
      
      newpos_x = newpos_x + nextisect_t * vec_x;
      newpos_y = newpos_y + nextisect_t * vec_y;
      
      gridpos_x = floor((newpos_x-x_bnd0)/dx);
      gridpos_y = floor((newpos_y-y_bnd0)/dy);
      
      if (gridpos_x >= 0.0f && gridpos_x < ((float)(nxbnd-1)) && gridpos_y >= 0.0f && gridpos_y < ((float)(nybnd-1))) {
	// photon reaches active area of detector
	//printf("Photon reaches detector\n");
	// increment photon onto detector
	
	// use OpenCL atomic_inc() function so that different compute threads
	// writing to the same detector do not trip over each other
	atomic_inc(&detector_photons[ (nybnd-1)*((unsigned)gridpos_x) + ((unsigned)gridpos_y) ]);
      } else {
	// The photon reached the detector plane, but not within 
	// the detector area
	
	//printf("Photon off detector area\n");
      }
      
      
      photon_gone=true; 
    }
    
    
    // Check for photon leaving our domain
    
    if (newpos_xbndidx <= 0.0f || newpos_xbndidx >= ((float)nxbnd)-1.0f) {
      //printf("Photon exited along y axis\n");
      photon_gone=true; // photon left across x
    }
    if (newpos_ybndidx <= -0.0f || newpos_ybndidx >= ((float)nybnd)-1.0f) {
      //printf("Photon exited along x axis\n");
      photon_gone=true; // photon left across x
    }
    if (newpos_zbndidx <= 0.0f && newvec_z <= 0.0f) {
      //printf("Photon backscattered\n");
      photon_gone=true; // photon backscattered toward source
    }
    
    

    // Check for conditions that shouldn't happen
    // (especially not very often!)
    if (itercnt > iterlimit) {
      // ***!!! DO NOT COMMENT OUT THIS PRINT STATEMENT
      printf("Max itercnt exceeded: photon @ (%f,%f,%f) vector (%f,%f,%f)\n",newpos_x,newpos_y,newpos_z,newvec_x,newvec_y,newvec_z);
      photon_gone=true;
    }
    
    if (isnan(newvec_x) || isnan(newvec_y) || isnan(newvec_z)) {
      // ***!!! DO NOT COMMENT OUT THIS PRINT STATEMENT
      printf("NaN photon destroyed\n");
      photon_gone=true;
    }
    
    if (!isfinite(newvec_x) || !isfinite(newvec_y) || !isfinite(newvec_z)) {
      // ***!!! DO NOT COMMENT OUT THIS PRINT STATEMENT
      printf("NaN photon destroyed\n");
      photon_gone=true;
    }
    
    if (fabs(pow(newvec_x,2.0f)+pow(newvec_y,2.0f)+pow(newvec_z,2.0f)-1.0f) > .01f) {
      // ***!!! DO NOT COMMENT OUT THIS PRINT STATEMENT
      printf("Photon propagation direction not correctly normalized; photon destroyed\n");
      photon_gone=true;
    }
      
    itercnt++;

    if (photon_gone) {
      need_new_photon=1; // get a new photon in the next iteration
    }
  
  }
}
    
    
