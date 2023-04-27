//
// Created by 30500 on 2022/2/14 0014.
//

#ifndef NCNN_FILE_ARM_H
#define NCNN_FILE_ARM_H
#include <csignal>
#include <fstream>
#include <vector>

extern FILE* arm_weight_file;
extern FILE* arm_weight_file_read;
extern FILE* arm_weight_file_reads[8] ;

extern int load_weight_flag;

extern int ARM_W_TEST;
extern int USE_PACK_ARM ;
extern int USE_KERNAL_ARM ;

extern std::vector<size_t> arm_weight_file_seek_Vectors;
extern size_t arm_weight_file_seek_save;


void arm_weight_file_init( const char* comment );

void arm_weight_file_read_init( const char* comment );

void WritearmWeightDataReaderFile( const char* comment);

void ReadarmWeightDataReaderFile( const char* comment);

#endif //NCNN_FILE_ARM_H
