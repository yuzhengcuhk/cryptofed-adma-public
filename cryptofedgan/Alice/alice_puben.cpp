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
    
    
    ifstream readpubkey("pubkey.dat", ios::binary);
    //ifstream readrelinkey("relinkey.dat", ios::binary);
    
    KeyGenerator keygen(context);          //SEAL are public key encryption schemes
    auto public_key = keygen.public_key();  //just initialize the key
    public_key.load(context, readpubkey);  //load publickey from txt file
    //auto relin_key = keygen.relin_keys();
    //relin_key.load(context, readrelinkey);
    
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    CKKSEncoder encoder(context);
    
    double test;
    ifstream data("ABsum.txt", ios::binary);
    ofstream en_data("alice_en.dat", ios::binary);
    Ciphertext x_encrypted;
    Plaintext x_plain_double;
    
    while (data >> test) {
        encoder.encode(test, scale, x_plain_double);
        encryptor.encrypt(x_plain_double, x_encrypted);
        //evaluator.relinearize_inplace(x_encrypted, relin_key);
        //evaluator.rescale_to_next_inplace(x_encrypted);
        x_encrypted.save(en_data);
        en_data << " ";
    }
    readpubkey.close();
    data.close();
    en_data.close();
    
    return 0;
}
