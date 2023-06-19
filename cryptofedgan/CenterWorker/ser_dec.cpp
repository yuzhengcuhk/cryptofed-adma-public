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
    
    KeyGenerator keygen(context);          //SEAL are public key encryption schemes
    
    ifstream readpubkey("pubkey.dat", ios::binary);
    ifstream readserkey("serkey.dat", ios::binary);
    auto public_key = keygen.public_key();
    auto secret_key = keygen.secret_key();
    public_key.load(context, readpubkey);
    secret_key.load(context, readserkey);
    //auto relin_keys = keygen.relin_keys();  no multiplication may not need relin_key
    
    
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);
    
    double bob, alice;
    ifstream bobdata("bob_en.dat", ios::binary);
    ifstream alicedata("alice_en.dat", ios::binary);
    
    ofstream tmpbobpara("tmp_bob.dat", ios::binary);
    ofstream tmpalicepara("tmp_alice.dat", ios::binary);
    ofstream en_data("decrypt_sum.txt", ios::binary);
    Ciphertext bob_encrypted, alice_encrypted, ab_result;

    Plaintext plain_result;
    
    int i = 0;
    int bobfilelen = 0;
    int tmpfilelen = 0;
    int alicefilelen = 0;
    
    bobdata.seekg(0, ios::end);             //locate the pointer to the end of file
    bobfilelen = bobdata.tellg();
    bobdata.seekg(0, ios::beg);
    
    //cout << sizeof(Ciphertext) << " " << bobfilelen << endl;
    while (i < bobfilelen) {
        bob_encrypted.load(context, bobdata);
        alice_encrypted.load(context, alicedata);
        //bob_encrypted.load(context, copybob);
        evaluator.add(bob_encrypted, alice_encrypted, ab_result);
        
        ofstream tmppara("tmp_para.dat", ios::binary);
        bob_encrypted.save(tmppara);
        tmppara << " ";
        tmppara.close();
        
        ifstream getlen("tmp_bob.dat");
        getlen.seekg(0, ios::end);
        tmpfilelen += getlen.tellg();
        
        ifstream alicelen("tmp_alice.dat");
        alicelen.seekg(0, ios::end);
        alicefilelen += alicelen.tellg();
        
        decryptor.decrypt(ab_result, plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        en_data << result[0] << " ";
        bobdata.seekg(tmpfilelen, ios::beg);
        alicedata.seekg(alicefilelen, ios::beg);
        i += tmpfilelen;
    }
    
    bobdata.close();
    alicedata.close();
    tmpbobpara.close();
    tmpalicepara.close();
    en_data.close();
    
    return 0;
}
