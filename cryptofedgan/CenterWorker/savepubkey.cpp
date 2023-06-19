//
//  main.cpp
//  EnPara_SealBFV
//
//  Created by Song on 2020/1/27.
//  Copyright © 2020 Song. All rights reserved.
//

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "seal/seal.h"

using namespace std;
using namespace seal;

int main(int argc, const char * argv[]) {
    //use bfv scheme
    EncryptionParameters parms(scheme_type::CKKS);
    //set the poly_modulus_degree
    size_t poly_modulus_degree = 4096;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {36, 36, 36}));
    
    double scale = pow(2.0, 40);
    
    auto context = SEALContext::Create(parms);  //construct a SEALContext object
    
    /*both publickey and secretkey need to be saved*/
    KeyGenerator keygen(context);          //SEAL are public key encryption schemes
    auto public_key = keygen.public_key();
    auto secret_key = keygen.secret_key();
    auto relin_key = keygen.relin_keys();
    
    ofstream pubkeyfile("pubkey.dat", ios::binary);
    ofstream secretkeyfile("serkey.dat", ios::binary);
    ofstream relkeyfile("relinkey.dat", ios::binary);
    
    public_key.save(pubkeyfile);
    secret_key.save(secretkeyfile);
    relin_key.save(relkeyfile);
    
    pubkeyfile.close();
    secretkeyfile.close();
    relkeyfile.close();
    
    return 0;
}
