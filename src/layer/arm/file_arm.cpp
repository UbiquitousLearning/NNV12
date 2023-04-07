//
// Created by 30500 on 2022/2/14 0014.
//

#include <iostream>
#include "file_arm.h"
FILE* arm_weight_file;
FILE* arm_weight_file_read;
FILE* arm_weight_file_reads[8] ;

std::vector<size_t> arm_weight_file_seek_Vectors;

int load_weight_flag = 1;

int ARM_W_TEST= 0;
int USE_PACK_ARM = 0;
int USE_KERNAL_ARM =0;

void arm_weight_file_init( const char* comment ){
    char path[256];
    sprintf(path,  "" "%s.arm.bin", comment);
    arm_weight_file = fopen(path, "wb");
    std::cout<<"[WRITE]["<<path<<"][file]arm_weight_file"<<std::endl;
}

void arm_weight_file_read_init( const char* comment ){
    char path[256];
    sprintf(path,  "" "%s.arm.bin", comment);
    std::cout<<"[READ-]["<<path<<"][file]arm_weight_file_reads[0~8]"<<std::endl;
    arm_weight_file_read = fopen(path, "rb");
    for(int i=0; i < 8; i++){
        //    for(FILE* fs: arm_weight_file_reads){
        arm_weight_file_reads[i] = fopen(path, "rb");
    }
}
size_t arm_weight_file_seek_save = 0;

void WritearmWeightDataReaderFile( const char* comment)
{

    char seek_path[256];
    sprintf(seek_path,  "" "%s.arm.br.dat", comment);
    //    arm_weight_file_seek = fopen(seek_path, "wb");

    std::cout<<"[WRITE]["<<seek_path<<"]arm_weight_file_seek_Vectors(len:"<<arm_weight_file_seek_Vectors.size()<<"):{";
    for(int i : arm_weight_file_seek_Vectors){
        printf("%d,", i);
    }
    std::cout<<"}"<<std::endl;

    std::ofstream osData(seek_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

    int arm_weight_file_seek_Size = arm_weight_file_seek_Vectors.size();
    osData.write(reinterpret_cast<char *>(&arm_weight_file_seek_Size), sizeof(arm_weight_file_seek_Size));
    osData.write(reinterpret_cast<char *>(arm_weight_file_seek_Vectors.data()), arm_weight_file_seek_Size*sizeof(arm_weight_file_seek_Vectors.front()) );

    osData.close();
}

void ReadarmWeightDataReaderFile( const char* comment)
{
    char path[256];
    sprintf(path,  "" "%s.arm.br.dat", comment);
    std::ifstream isData(path, std::ios_base::in | std::ios_base::binary);
    if (isData)
    {
        int arm_weight_file_seek_Size;
        isData.read(reinterpret_cast<char *>(&arm_weight_file_seek_Size),sizeof(arm_weight_file_seek_Size));
        arm_weight_file_seek_Vectors.resize(arm_weight_file_seek_Size);
        isData.read(reinterpret_cast<char *>(arm_weight_file_seek_Vectors.data()), arm_weight_file_seek_Size *sizeof(size_t) );

    }
    else
    {
        printf("ERROR: Cannot open file 数据.dat");
    }
    isData.close();
    std::cout<<"[READ-]["<<path<<"]arm_weight_file_seek_Vectors(len:"<<arm_weight_file_seek_Vectors.size()<<"):{";
    for(int i : arm_weight_file_seek_Vectors){
        printf("%d,", i);
    }
    std::cout<<"}"<<std::endl;
}