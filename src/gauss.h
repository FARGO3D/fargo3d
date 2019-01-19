for(ir=0; ir<NFLUIDS-1; ir++){ //for each equation
  
  //We now look for the max value in the column equal to the
  //current row and use it as pivot element
  max_value = -1;
  for(ir2=ir; ir2<NFLUIDS; ir2++){
    if( fabs(m[ir + ir2*NFLUIDS] ) >= max_value ){
      max_value = m[ir + ir2*NFLUIDS];
      ir_max    = ir2;
    }
  }
  
  //Rows interchange----------------------------------
  //We now interchange rows
  if (ir_max != ir) {
    for(ic=0;ic<NFLUIDS;ic++){
      temp = m[ic + ir*NFLUIDS];
      m[ic + ir*NFLUIDS] = m[ic + ir_max*NFLUIDS];
      m[ic + ir_max*NFLUIDS] = temp;
    }
    //and do the same for the vector b
    temp  = b[ir];
    b[ir] = b[ir_max];
    b[ir_max] = temp;
  }
  
  //-------------------------------------------------
  
  //We now perform the Gaussian elimination----------
  for(ir2=ir+1;ir2<NFLUIDS;ir2++){
    factor = m[ir + ir2*NFLUIDS]/m[ir+ir*NFLUIDS];
    for(ic=0; ic<NFLUIDS; ic++) m[ic + ir2*NFLUIDS] -= m[ic + ir*NFLUIDS]*factor;
    //and do the same for the vector b
    b[ir2] -= b[ir]*factor;
  }
 }
//--------------------------------------------------

// We now perform the back substitution-----------------
for(ir=NFLUIDS-1; ir>-1 ;ir--){
  for(ic=NFLUIDS-1; ic>ir ;ic--){
    b[ir] -= m[ic+ir*NFLUIDS]*b[ic];
  }
  b[ir] /= m[ir+ir*NFLUIDS];
 }
