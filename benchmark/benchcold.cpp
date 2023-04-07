// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <float.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <algorithm>
#include <iostream>
#include <utility>
#include <thread>
#include <chrono>
#include <functional>
#include <atomic>
#include <sys/time.h>
#include <omp.h>
#include <fstream>

#ifdef _WIN32
#include <algorithm>
#include <windows.h> // Sleep()
#else
#include <unistd.h> // sleep()
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net_new.h"
#include "gpu.h"
#include "pipeline.h"
#include "layer/arm/file_arm.h"

#define __ARM_NEON 0
#define __CPU_MASK__ 1
#if __ARM_NEON
#include <arm_neon.h>
#endif

//
//export GOMP_CPU_AFFINITY="7 6 5 4";

char globalbinpath[256];
char gloableparampath[256];

double start_t = 0;

//ncnn::Net net;
#ifdef __EMSCRIPTEN__
#define MODEL_DIR "/working/"
#else
#define MODEL_DIR ""
#endif

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* format, void* p) const
    {
        return 0;
    }
    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

static int g_warmup_loop_count = 8;
static int g_loop_count = 4;
static bool g_enable_cooling_down = true;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

#if NCNN_VULKAN
static ncnn::VulkanDevice* g_vkdev = 0;
static ncnn::VkAllocator* g_blob_vkallocator = 0;
static ncnn::VkAllocator* g_staging_vkallocator = 0;
#endif // NCNN_VULKAN

//#define CPU_NUMS 4
//int CPU_LITTLE_s[CPU_NUMS] = {0, 1, 2, 3};
int CPU_NUMS;// = 4;
std::vector<int> CPU_LITTLE_s;// = {0, 1, 2, 3};
int CPU_BIG_;// = 7;
//std::vector<int> cpuLittle_Vectors[CPU_NUMS];
std::vector<std::vector<int>> cpuLittle_Vectors;//(CPU_NUMS);
std::vector<int> cpuBig_Vector;

void setDeviceConfig(int device_cpus){
    if(device_cpus == 0){
        CPU_NUMS= 4;
        CPU_LITTLE_s.resize(CPU_NUMS);
        CPU_LITTLE_s = {0, 1, 2, 3};
        CPU_BIG_ = 7;
        cpuLittle_Vectors.resize(CPU_NUMS);
    }
    else if(device_cpus == 1){
        CPU_NUMS= 2;
        CPU_LITTLE_s.resize(CPU_NUMS);
        CPU_LITTLE_s = {1, 2};
        CPU_BIG_ = 0;
        cpuLittle_Vectors.resize(CPU_NUMS);
    }
    else{
        CPU_NUMS= 4;
        CPU_LITTLE_s.resize(CPU_NUMS);
        CPU_LITTLE_s = {0, 1, 2, 3};
        CPU_BIG_ = 7;
        cpuLittle_Vectors.resize(CPU_NUMS);
    }
}

////////////////////////files//////////////////////////////////

void ReadBinaryFile( const char* comment, bool use_vulkan_compute)
{
    double start = ncnn::get_current_time();

    char path[256];
    if(use_vulkan_compute){
        sprintf(path, MODEL_DIR "%s.vk.dat", comment);
    }
    else{
        sprintf(path, MODEL_DIR "%s.dat", comment);
    }
    std::ifstream isData(path, std::ios_base::in | std::ios_base::binary);
    if (isData)
    {
        for (int i=0; i<CPU_NUMS; i++){
            int n = CPU_NUMS - i-1;
            int cpu_Vsize;
            isData.read(reinterpret_cast<char *>(&cpu_Vsize),sizeof(cpu_Vsize));
            cpuLittle_Vectors[n].resize(cpu_Vsize);
            isData.read(reinterpret_cast<char *>(cpuLittle_Vectors[n].data()), cpu_Vsize *sizeof(int) );

            printf("%d:{", n);
            for(int i : cpuLittle_Vectors[n]){
                printf("%d,", i);
            }
            printf("}\n");
        }
        int cpu7Vsize;
        isData.read(reinterpret_cast<char *>(&cpu7Vsize),sizeof(cpu7Vsize));
        cpuBig_Vector.resize(cpu7Vsize);
        isData.read(reinterpret_cast<char *>(cpuBig_Vector.data()), cpu7Vsize *sizeof(int) );

        printf("7:{");
        for(int i : cpuBig_Vector){
            printf("%d,", i);
        }
        printf("}\n");
    }
    else
    {
        printf("ERROR: Cannot open file ����.dat");
    }
    double e = ncnn::get_current_time();
    printf("rfile_time %f\n", e-start);
    isData.close();
}
void WriteDataReaderFile( const char* comment)
{

    printf("DR_file_Vectors:{");
    for(int i : DR_file_Vectors){
        printf("%d,", i);
    }
    printf("}\n");

    char path[256];
    sprintf(path, MODEL_DIR "%s.br.dat", comment);
    std::ofstream osData(path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

    int DRListSize = DR_file_Vectors.size();
    osData.write(reinterpret_cast<char *>(&DRListSize), sizeof(DRListSize));
    osData.write(reinterpret_cast<char *>(DR_file_Vectors.data()), DRListSize*sizeof(DR_file_Vectors.front()) );

    osData.close();
}
void ReadDataReaderFile( const char* comment)
{
    char path[256];
    sprintf(path, MODEL_DIR "%s.br.dat", comment);
    std::ifstream isData(path, std::ios_base::in | std::ios_base::binary);
    if (isData)
    {
        int DRListSize;
        isData.read(reinterpret_cast<char *>(&DRListSize),sizeof(DRListSize));
        DR_file_Vectors.resize(DRListSize);
        isData.read(reinterpret_cast<char *>(DR_file_Vectors.data()), DRListSize *sizeof(size_t) );

    }
    else
    {
        printf("ERROR: Cannot open file ����.dat");
    }
    isData.close();
    printf("DR_file_Vectors:{");
    for(int i : DR_file_Vectors){
        printf("%d,", i);
    }
    printf("}\n");
}
void WriteSprivMapBinaryFile( const char* comment)
{
    char path[256];
    sprintf(path, MODEL_DIR "%s.m.dat", comment);
    std::ofstream osData(path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

    int ssSize = ncnn::spriv_map.size();
    osData.write(reinterpret_cast<char *>(&ssSize), sizeof(ssSize));
    for(auto &it:ncnn::spriv_map)
    {
        auto k = it.first;
        osData.write(reinterpret_cast<char *>(&k), sizeof(k));
        auto s = it.second;
        int sSize = s.size();
        osData.write(reinterpret_cast<char *>(&sSize), sizeof(sSize));
        osData.write(reinterpret_cast<char *>(s.data()), sSize*sizeof(s.front()) );
        //        cout<<it.first<<" "<<it.second<<endl;
    }
    osData.close();
}
void ReadSprivMapBinaryFile( const char* comment)
{

    double start = ncnn::get_current_time();
    char path[256];
    sprintf(path, MODEL_DIR "%s.m.dat", comment);
    std::ifstream isData(path, std::ios_base::in | std::ios_base::binary);
    int ssSize;
    isData.read(reinterpret_cast<char *>(&ssSize),sizeof(ssSize));
    if (isData)
    {
        for(int i=0; i<ssSize; i++)
        {
            int key;
            isData.read(reinterpret_cast<char*>(&key), sizeof(key));
            int sSize;
            isData.read(reinterpret_cast<char*>(&sSize), sizeof(sSize));
            std::vector<uint32_t> spirv;
            spirv.resize(sSize);
            isData.read(reinterpret_cast<char*>(spirv.data()), sSize * sizeof(uint32_t));
            ncnn::spriv_map.insert(std::pair<int, std::vector<uint32_t>>(key, spirv));
            //            ncnn::spriv_vectors.push_back(spriv);
            //            printf("%d  %d:{", key, sSize);
            ////            for(int i : spirv){
            ////                printf("%d,", i);
            ////            }
            //            printf("}\n");
        }
    }
    else
    {
        printf("ERROR: Cannot open file ����.m.dat");
    }

    double e = ncnn::get_current_time();
    printf("rmap_time %f\n", e-start);
    isData.close();
}

double cold_latency;
void WriteOutputFile(const char* comment){
    char path[256];
    sprintf(path, "%s, ", comment);

    std::ofstream out;

    char src[50];
    strcpy(src,  comment);
    int len = strlen(src);
    char * last = const_cast<char*>(strrchr(src, '/') + 0);
    if(last != NULL){
        *(last+1)= '\0';
        strcat(src, "output.csv");
        out.open(src, std::ios::out | std::ios::app);  //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
    }else{
        out.open("output.csv", std::ios::out | std::ios::app);  //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
    }
    if (!out.is_open())
    {
//        printf("open file output.csv error");
        std::cout<<"[ERROR] open file: "<<src<<std::endl;
        return;
    }
    out<<path<<cold_latency<<"ms"<<std::endl;
    std::cout<<"write file: "<<src<<std::endl;
    out.close();
}

////////////////////////benchmark//////////////////////////////////

void benchmark(const char* comment, const ncnn::Mat& _in, const ncnn::Option& opt)
{
    ncnn::Mat in = _in;
    in.fill(0.01f);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        g_blob_vkallocator->clear();
        g_staging_vkallocator->clear();
    }
#endif // NCNN_VULKAN

    ncnn::Net net;

    net.opt = opt;

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        net.set_vulkan_device(g_vkdev);
    }
#endif // NCNN_VULKAN

#ifdef __EMSCRIPTEN__
#define MODEL_DIR "/working/"
#else
#define MODEL_DIR ""
#endif

    char parampath[256];
    sprintf(parampath, MODEL_DIR "%s.param", comment);

    char binpath[256];
    sprintf(binpath, MODEL_DIR "%s.bin", comment);

    double load_param_start = ncnn::get_current_time();
    net.load_param(parampath);
    double load_param_end = ncnn::get_current_time();
    double load_param_time = load_param_end - load_param_start;
    arm_weight_file_seek_Vectors.resize(net.layers().size());

    double load_model_start = ncnn::get_current_time();

    int open = net.load_model_dr(binpath);
    if (open<0)
    {
        DataReaderFromEmpty dr;
        net.load_model_dr(dr);
    }
    double load_model_end = ncnn::get_current_time();
    double load_model_time = load_model_end - load_model_start;


    net.load_model_pipe();
    double load_pipe_end = ncnn::get_current_time();
    double load_pipe_time = load_pipe_end - load_model_end;
    //    DataReaderFromEmpty dr;
    //    net.load_model(dr);

    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    if (g_enable_cooling_down)
    {
        // sleep 10 seconds for cooling down SOC  :(
#ifdef _WIN32
        Sleep(10 * 1000);
#elif defined(__unix__) || defined(__APPLE__)
        // sleep(10);
#elif _POSIX_TIMERS
        struct timespec ts;
        ts.tv_sec = 10;
        ts.tv_nsec = 0;
        nanosleep(&ts, &ts);
#else
        // TODO How to handle it ?
#endif
    }

    ncnn::Mat out;

    // warm up
    double first_time;
    for (int i = 0; i < 1; i++)
    {
        double start = ncnn::get_current_time();

        ncnn::Extractor ex = net.create_extractor();
        ex.input(input_names[0], in);
        ex.extract(output_names[0], out);

        double end = ncnn::get_current_time();
        first_time = end - start;
//        if (i==0){
//            first_time = time;
//        }
//                fprintf(stderr, "first %f\n", for_cal_time);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i = 0; i < g_loop_count; i++)
    {
        for_cal_time = 0;
        double start = ncnn::get_current_time();

        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input(input_names[0], in);
            ex.extract(output_names[0], out);
                        float* ptr = (float*)out.data;
                        printf("out %f\n", *ptr);
//                        fprintf(stderr, "%f\n", for_cal_time);
        }

        double end = ncnn::get_current_time();

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
        //        fprintf(stderr, "%f\n", time);
    }

    time_avg /= g_loop_count;

    fprintf(stderr, "%20s  load param = %7.2f  load model = %7.2f  create pipeline = %7.2f  first = %7.2f  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, load_param_time, load_model_time, load_pipe_time, first_time, time_min, time_max, time_avg);
}

////////////////////////benchmark_new/////////////////////////////

void inference(const ncnn::Net& net, const ncnn::Mat& in, bool print=true){
    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    if (g_enable_cooling_down)
    {
        // sleep 10 seconds for cooling down SOC  :(
#ifdef _WIN32
        Sleep(10 * 1000);
#elif defined(__unix__) || defined(__APPLE__)
//    sleep(10);
#elif _POSIX_TIMERS
        struct timespec ts;
        ts.tv_sec = 10;
        ts.tv_nsec = 0;
        nanosleep(&ts, &ts);
#else
// TODO How to handle it ?
#endif
    }

    ncnn::Mat out;

    //    infer_start = ncnn::get_current_time();

    ncnn::Extractor ex = net.create_extractor();
    ex.input(input_names[0], in);
    ex.extract(output_names[0], out);

    float* ptr = (float*)out.data;
    infer_end = ncnn::get_current_time();
    infer_time = for_cal_time;//infer_end - infer_start;
    if(print)
    {
        printf("_____    out[0] = %f", *ptr);
        printf("_____    for_skp_time=%f for_cal_time=%f\t", for_skp_time, for_cal_time);
        //        fprintf(stderr, "infer time =  %7.2f\n", infer_time);
        printf("_____    infer time =  %7.2f\n", infer_time);
    }

    //    pthread_exit(&t_infer);

}

typedef struct net_cid{
    int cpu_id;
    int cpu_set;
    ncnn::Net* net;
//    ncnn::ModelBinFromDataReader* mb;
}net_cid;
typedef struct net_cid_f{
    int cpu_id;
    int cpu_set;
    ncnn::Net* net;
//    ncnn::ModelBinFromDataReader* mb;
//    FILE * f;
} net_cid_file;
void *thread_list(void *args){

    auto *net_cid_ = (net_cid*)args;

#ifdef __CPU_MASK__
    cpu_set_t mask;  //CPU�˵ļ���
    CPU_ZERO(&mask);    //�ÿ�
    CPU_SET(net_cid_->cpu_set,&mask);   //�����׺���ֵ
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//�����߳�CPU�׺���
    {
        printf("warning: could not set CPU affinity, continuing...\n");
    }
#endif
//    printf("__________t%d_cpu%d    tid=%ld,cpu=%d\n",net_cid_->cpu_id, net_cid_->cpu_set, pthread_self(), sched_getcpu());
    printf("_____    Thread_load&pipe_(cpu%d)([%d])    tid=%ld,cpu=%d\n",net_cid_->cpu_set,net_cid_->cpu_id,  pthread_self(), sched_getcpu());
    double start_time = ncnn::get_current_time();

    DataReaderFromEmpty dr;
    ncnn::ModelBinFromDataReader mb(dr);//get_dr(fp, net));

    //        auto net=net_cid_->net;
    int list_len = cpuLittle_Vectors[net_cid_->cpu_id].size();
    double cal_time= 0;
    for(int ii=0; ii<list_len; ii++){
        int i = cpuLittle_Vectors[net_cid_->cpu_id][ii];
        double s = ncnn::get_current_time();
        net_cid_->net->load_model_layer(mb, i);
        if(USE_PACK_ARM)
        {
            fseek(arm_weight_file_reads[sched_getcpu()], arm_weight_file_seek_Vectors[i - 1], SEEK_SET);
        }
        net_cid_->net->load_pipe_layer(i);
        double e = ncnn::get_current_time();

        cal_time+= e-s;
        if(timeshow)
        {
            printf("%d,%f\n", i, e - s);
        }
        //        net->load_model_layer(mb, i);
        //        net->load_pipe_layer(i);
        syn_act(infer_syn, i);
    }
    double end_time = ncnn::get_current_time();
//    printf("\n==============================================================================================list_cpu%d_time = %f %f\n", sched_getcpu(), end_time - start_time, cal_time);
    printf("_____    Thread_load&pipe_(cpu%d)([%d])[FINISH]        time = %f %f\n", net_cid_->cpu_set,sched_getcpu(), end_time - start_time, cal_time);

    return 0;
}

void * thread_list_file(void *args){

    auto *net_cid_ = (net_cid_file*)args;
#ifdef __CPU_MASK__
    cpu_set_t mask;  //CPU�˵ļ���
    CPU_ZERO(&mask);    //�ÿ�
    CPU_SET(net_cid_->cpu_set,&mask);   //�����׺���ֵ
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//�����߳�CPU�׺���
    {
        printf("warning: could not set CPU affinity, continuing...\n");
    }
#endif
    FILE* fp_ = fopen(globalbinpath, "rb");
    fseek(fp_,0,SEEK_SET);
    ncnn::DataReaderFromStdio dr_(fp_);
    ncnn::ModelBinFromDataReader mb_(dr_);

//#if NCNN_VULKAN
//    FILE*  vk_weight_file_read_ = fopen(vk_weight_file_read_path, "rb");
//    fseek(vk_weight_file_read_,  0, SEEK_SET);
//#endif

//    printf("__________t%d_cpu%d    tid=%ld,cpu=%d\n",net_cid_->cpu_id, net_cid_->cpu_set, pthread_self(), sched_getcpu());
    printf("_____    Thread_load&pipe_(cpu%d)([%d])    tid=%ld,cpu=%d\n",net_cid_->cpu_set,net_cid_->cpu_id,  pthread_self(), sched_getcpu());
    double start_time = ncnn::get_current_time();


    //        auto net=net_cid_->net;
    int list_len = cpuLittle_Vectors[net_cid_->cpu_id].size();
    double cal_time= 0;
    for(int ii=0; ii<list_len; ii++){
        int i = cpuLittle_Vectors[net_cid_->cpu_id][ii];
        double s = ncnn::get_current_time();

        fseek(fp_,  DR_file_Vectors[i], SEEK_SET);
        net_cid_->net->load_model_layer(mb_, i);
        if(USE_PACK_ARM)
        {
            fseek(arm_weight_file_reads[sched_getcpu()], arm_weight_file_seek_Vectors[i - 1], SEEK_SET);
        }
        net_cid_->net->load_pipe_layer(i);
//#if NCNN_VULKAN
//        fseek(vk_weight_file_read_,  vk_weight_file_seek_Vectors[i], SEEK_SET);
//#endif
        net_cid_->net->upload_model_layer(i);
        double e = ncnn::get_current_time();

        cal_time+= e-s;
        if(timeshow)
        {
            printf("%d,%f\n", i, e - s);
        }
        //        net->load_model_layer(mb, i);
        //        net->load_pipe_layer(i);
        syn_act(infer_syn, i);
    }
    double end_time = ncnn::get_current_time();
//    printf("\n==============================================================================================list_cpu%d_time = %f %f\n", sched_getcpu(), end_time - start_time, cal_time);
    printf("_____    Thread_load&pipe_(cpu%d)([%d])[FINISH]        time = %f %f\n", net_cid_->cpu_set,sched_getcpu(), end_time - start_time, cal_time);

    if (fp_)
    {
        fclose(fp_);
    }
    return 0;
}

void cold_boot_empty(ncnn::Net& net, const ncnn::Mat& in)
{
    printf("_____    Thread_exec_(cpu4567)     tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    DataReaderFromEmpty dr;
    ncnn::ModelBinFromDataReader mb(dr);//get_dr(fp, net));

    std::thread th[CPU_NUMS];
    net_cid net_cids[CPU_NUMS];
    for(int i = 0; i<CPU_NUMS; i++){
        net_cids[i] =  {i, CPU_LITTLE_s[i], &net};//, &mb};
        th[i] = std::thread(thread_list, (void*)&net_cids[i]);
    }
//    net_cid net_cid0 =  {0, 1, &net, &mb};
//    std::thread t_0(thread_list, (void*)&net_cid0);
//    net_cid net_cid1 =  {1, 2, &net, &mb};
//    std::thread t_1(thread_list, (void*)&net_cid1);

    /*create*/
    double s = ncnn::get_current_time();
    net.load_model_layer(mb, 0);
    net.load_pipe_layer(0);
    net.load_model_layer(mb, 1);
    syn_act(read_syn, 2);
    net.load_pipe_layer(1);
    syn_act(infer_syn, 1);
    int list_len = cpuBig_Vector.size();
    double cal_time= 0;
    for(int ii=0; ii<list_len; ii++){
        int i = cpuBig_Vector[ii];
        net.load_model_layer(mb, i);
        if(USE_PACK_ARM)
        {
            fseek(arm_weight_file_reads[sched_getcpu()], arm_weight_file_seek_Vectors[i - 1], SEEK_SET);
        }
        net.load_pipe_layer(i);
        syn_act(infer_syn, i);
    }

//    for(int i = 0; i<CPU_NUMS; i++){
//        th[i].join();
//    }
    double start_time_inf = ncnn::get_current_time();
    inference(net, in, true);
    double end_time = ncnn::get_current_time();
    printf("_____    Thread_exec_(cpu4567)[FINISH]     rc=%f infer=%f \n", start_time_inf-s, end_time - start_time_inf);
//    printf("\n===================================================================================================rc=%f infer=%f \n", start_time_inf-s, end_time - start_time_inf);
//    t_0.join();
//    t_1.join();
    for(int i = 0; i<CPU_NUMS; i++){
        th[i].join();
    }
}

void cold_boot_file(ncnn::Net &net, FILE* fp, const ncnn::Mat& in){
    printf("_____    Thread_exec_(cpu4567)     tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    ncnn::DataReaderFromStdio dr(fp);
    ncnn::ModelBinFromDataReader mb(dr);

    FILE* fps[CPU_NUMS];
    std::thread th[CPU_NUMS];
    net_cid_file net_cids[CPU_NUMS];
    ncnn::ModelBinFromDataReader mb_(mb);//[thread_times];//;

    for(int i = 0; i<CPU_NUMS; i++){
        fps[i] = fopen(globalbinpath, "rb");
        fseek(fps[i],0,SEEK_SET);
        net_cids[i] =  {i, CPU_LITTLE_s[i], &net};//, &mb, fp};
        th[i] = std::thread(thread_list_file, (void*)&net_cids[i]);
    }


    double s = ncnn::get_current_time();
    fseek(fp, DR_file_Vectors[0], SEEK_SET);
    net.load_model_layer(mb, 0);
    net.load_pipe_layer(0);
//#if NCNN_VULKAN
//    fseek(vk_weight_file_read,  vk_weight_file_seek_Vectors[0], SEEK_SET);
//#endif
    net.upload_model_layer(0);
    fseek(fp, DR_file_Vectors[1], SEEK_SET);
    net.load_model_layer(mb, 1);
    syn_act(read_syn, 2);
    net.load_pipe_layer(1);
//#if NCNN_VULKAN
//    fseek(vk_weight_file_read,  vk_weight_file_seek_Vectors[1], SEEK_SET);
//#endif
    net.upload_model_layer(1);
    syn_act(infer_syn, 1);
    int list_len = cpuBig_Vector.size();
    double cal_time= 0;
    for(int ii=0; ii<list_len; ii++){
        int i = cpuBig_Vector[ii];
        fseek(fp, DR_file_Vectors[i], SEEK_SET);
        net.load_model_layer(mb, i);
        if(USE_PACK_ARM)
        {
            fseek(arm_weight_file_reads[sched_getcpu()], arm_weight_file_seek_Vectors[i - 1], SEEK_SET);
        }
        net.load_pipe_layer(i);
//#if NCNN_VULKAN
//        fseek(vk_weight_file_read,  vk_weight_file_seek_Vectors[i], SEEK_SET);
//#endif
        net.upload_model_layer(i);
        syn_act(infer_syn, i);
    }
//    for(int i = 0; i<CPU_NUMS; i++){
//        th[i].join();
//    }
//    double start_ul_inf = ncnn::get_current_time();
//    net.upload_models();
    double start_time_inf = ncnn::get_current_time();
    inference(net, in, true);
    double end_time = ncnn::get_current_time();
    printf("_____    Thread_exec_(cpu4567)[FINISH]     rc=%f infer=%f \n", start_time_inf-s, end_time - start_time_inf);
//    printf("\n===================================================================================================pipeline=%f infer=%f\n", start_time_inf-s, end_time - start_time_inf);
    //            t_0.join();
    //            t_1.join();
    for(int i = 0; i<CPU_NUMS; i++){
        th[i].join();
    }
    if (fp)
    {
        fclose(fp);
    }
}

int warmUp = 1;
void benchmark_new(const char* comment, const ncnn::Mat& _in, const ncnn::Option& opt)
{
    ncnn::Mat in = _in;
    in.fill(0.01f);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        g_blob_vkallocator->clear();
        g_staging_vkallocator->clear();
    }
#endif // NCNN_VULKAN

    ncnn::Net net;

    net.opt = opt;

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        net.set_vulkan_device(g_vkdev);
    }
#endif // NCNN_VULKAN

#ifdef __EMSCRIPTEN__
#define MODEL_DIR "/working/"
#else
#define MODEL_DIR ""
#endif

    char parampath[256];
    sprintf(parampath, MODEL_DIR "%s.param", comment);
    sprintf(gloableparampath, MODEL_DIR "%s.param", comment);

    char binpath[256];
    sprintf(binpath, MODEL_DIR "%s.bin", comment);
    sprintf(globalbinpath, MODEL_DIR "%s.bin", comment);


    ncnn::current_layer_idx_f2p = 0;
    ncnn::current_layer_idx_p2i = 0;


    net.load_param(parampath);
//    printf("ssssssssssssssss  %d\n", net.layers().size());



    printf("_____    Thread_exec_(cpu4567)     start %f\n", ncnn::get_current_time()-start_t);
//    printf("\n==s====================\n");
//    printf("\n=========s=============\n");
    double s_time = ncnn::get_current_time();

//    warmUp =0;
    if(warmUp) //Warm up SoC to max the SoC CPU's freq.
    {
        ncnn::Net net_in;
        net_in.opt = net.opt;
        char tparampath[256];
        sprintf(tparampath, MODEL_DIR "mobilenet_v3.param"); //GoogleNet  //AlexNet
        char tbinpath[256];
        sprintf(tbinpath, MODEL_DIR "mobilenet_v3.bin");
        net_in.load_param(gloableparampath);
        int open = net_in.load_model_dr(globalbinpath);
        if (open < 0)
        {
//            printf("_____    load file files\n");
            DataReaderFromEmpty dr;
            net_in.load_model_dr(dr);
        }
        net_in.load_model_pipe();
        //    printf("load p end %f\n", ncnn::get_current_time()-start_t);
        for (int i = 0; i < 30; i++)
        {
            ncnn::Mat top;
            //        conv7x7s2_pack1to4_neon(ncnn::Mat(227, 227, 3), top, ncnn::Mat(7,7, 3), ncnn::Mat(7,7, 3), net->opt);
            inference(net_in, ncnn::Mat(227, 227, 3), false);
            //        ncnn::do_forward_layer

            //        printf("infer p end %f\n", ncnn::get_current_time()-start_t);
        }
        ncnn::current_layer_idx_f2p = 0;
        ncnn::current_layer_idx_p2i = 0;
        infer_start = 0;
        pipe_start = 0;
        for_cal_time = 0;
        for_skp_time = 0;
        read_syn = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, {}};
        create_syn = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, {}};
        infer_syn = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, {}};
    }
    clear_times_save();
    resize_times_save(net.layers().size());

    //        printf("param finish_____________\n");
    printf("_____    Thread_exec_(cpu4567)     tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    if(opt.use_vulkan_compute)
    {
#ifdef __CPU_MASK__
        cpu_set_t mask;                                      //CPU�˵ļ���
        CPU_ZERO(&mask);                                     //�ÿ�
        CPU_SET(CPU_BIG_, &mask);                            //�����׺���ֵ
        if (sched_setaffinity(0, sizeof(mask), &mask) == -1) //�����߳�CPU�׺���
        {
            printf("warning: could not set CPU affinity, continuing...\n");
        }
#endif
    }
    //        printf("__________t_cpu4567 tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    double start_time = ncnn::get_current_time();
    save_start_time = ncnn::get_current_time();
    //read
    printf("_____    load file %s\n", globalbinpath);
    FILE* fp = fopen(globalbinpath, "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen %s failed", globalbinpath);
        cold_boot_empty(net, in);
    }
    else
    {
        if (net.layers().empty())
        {
            NCNN_LOGE("network graph not ready");
        }
        cold_boot_file(net, fp, in);
    }

//    printf("_____________________________________________________________________________total time %f\n", ncnn::get_current_time()-start_time);
//
//    printf("_____________________________________________________________________________total time + warmup %f\n", ncnn::get_current_time()-s_time);
//    printf("_____________________________________________________________________________real total time %f\n", ncnn::get_current_time()-start_t);
//    printf("==============================================\n");

    cold_latency =  ncnn::get_current_time()-start_time;
    printf("=============benchmark_new[FINISH]___total_time=%f=====total_time_+_warmup=%f___real_total_time=%f ==========\n", cold_latency, ncnn::get_current_time()-s_time, ncnn::get_current_time()-start_t);

    for(int ii=0; ii<0;ii++)
    {
        const std::vector<const char*>& input_names__ = net.input_names();
        const std::vector<const char*>& output_names__ = net.output_names();

        ncnn::Mat out__;

        infer_start = ncnn::get_current_time();

        ncnn::Extractor ex__ = net.create_extractor();
        ex__.input(input_names__[0], in);
        ex__.extract(output_names__[0], out__);

        float* ptr = (float*)out__.data;
        printf("%f\n", *ptr);

        infer_end = ncnn::get_current_time();
        infer_time = infer_end - infer_start;
        fprintf(stderr, "2d infer time =  %7.2f\n", infer_time);
    }
}


int main(int argc, char** argv)
{
    start_t = ncnn::get_current_time();
    int loop_count = 4;
    int num_threads = ncnn::get_cpu_count();
    int powersave = 0;
    int gpu_device = -1;
    int cooling_down = 1;

//    if (argc >= 2)
//    {
//        loop_count = atoi(argv[1]);
//    }
//    if (argc >= 3)
//    {
//        num_threads = atoi(argv[2]);
//    }
//    if (argc >= 4)
//    {
//        powersave = atoi(argv[3]);
//    }
//    if (argc >= 5)
//    {
//        gpu_device = atoi(argv[4]);
//    }
//    if (argc >= 6)
//    {
//        cooling_down = atoi(argv[5]);
//    }

    char* model_name;
    int device_cpus = 0;
    loop_count =1;
    if (argc >= 2)
    {
        model_name = argv[1];
    }
    if (argc >= 3)
    {
        num_threads = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        powersave = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        gpu_device = atoi(argv[4]);
    }
    if (argc >= 6)
    {
        cooling_down = atoi(argv[5]);
    }
    if (argc >= 7)
    {
        device_cpus = atoi(argv[6]);
    }

#ifdef __EMSCRIPTEN__
    EM_ASM(
        FS.mkdir('/working');
        FS.mount(NODEFS, {root: '.'}, '/working'););
#endif // __EMSCRIPTEN__

    bool use_vulkan_compute = gpu_device != -1;

    g_enable_cooling_down = cooling_down != 0;

    g_loop_count = loop_count;

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

#if NCNN_VULKAN

    double startt_t = ncnn::get_current_time();
    if (use_vulkan_compute)
    {
        g_warmup_loop_count = 10;

        g_vkdev = ncnn::get_gpu_device(gpu_device);

        g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
        g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
    }
    double end_t = ncnn::get_current_time();
    printf("NCNN_VULKAN init   %f\n",end_t -startt_t);
#endif // NCNN_VULKAN

    // default option
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
#if NCNN_VULKAN
    opt.blob_vkallocator = g_blob_vkallocator;
    opt.workspace_vkallocator = g_blob_vkallocator;
    opt.staging_vkallocator = g_staging_vkallocator;
#endif // NCNN_VULKAN
       //    opt.use_winograd_convolution = true;
       //    opt.use_sgemm_convolution = true;
       //    opt.use_int8_inference = true;
       //    opt.use_vulkan_compute = use_vulkan_compute;
       //    opt.use_fp16_packed = true;
       //    opt.use_fp16_storage = true;
       //    opt.use_fp16_arithmetic = true;
       //    opt.use_int8_storage = true;
       //    opt.use_int8_arithmetic = true;
       //    opt.use_packing_layout = true;
       //    opt.use_shader_pack8 = false;
       //    opt.use_image_storage = false;

    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = true;
    opt.use_vulkan_compute = use_vulkan_compute;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = true;
    opt.use_packing_layout = true;
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "cooling_down = %d\n", (int)g_enable_cooling_down);
    setDeviceConfig(device_cpus);

    ncnn::Mat in = ncnn::Mat(224, 224, 3);

//    char model_name[] = "alexnet";
//    ncnn::Mat in = ncnn::Mat(227, 227, 3);

//    char model_name[] = "googlenet";
//    ncnn::Mat in = ncnn::Mat(224, 224, 3);

//    char model_name[] = "MobileNet";
//    ncnn::Mat in = ncnn::Mat(224, 224, 3);

//    char model_name[] = "MobileNetV2";
//    ncnn::Mat in = ncnn::Mat(224, 224, 3);

//    char model_name[] = "resnet18";
//    ncnn::Mat in = ncnn::Mat(224, 224, 3);

//    char model_name[] = "shufflenet";
//    ncnn::Mat in = ncnn::Mat(224, 224, 3);

//    char model_name[] = "efficientnet_b0";
//    ncnn::Mat in = ncnn::Mat(224, 224, 3);

//    char model_name[] = "resnet50";
//    ncnn::Mat in = ncnn::Mat(224, 224, 3);

//    char model_name[] = "squeezenet";
//    ncnn::Mat in = ncnn::Mat(227, 227, 3);

//    char model_name[] = "shufflenet_v2";
//    ncnn::Mat in = ncnn::Mat(224, 224, 3);

//    char model_name[] = "yolov4-tiny";
//    ncnn::Mat in = ncnn::Mat(416, 416, 3);

//    char model_name[] = "mobilenetv2_yolov3";
//    ncnn::Mat in = ncnn::Mat(352, 352, 3);

//    char model_name[] = "mobilenet_yolo";
//    ncnn::Mat in = ncnn::Mat(416, 416, 3);

    ReadDataReaderFile(model_name); //load_model(�� file read ˳��
    if (use_vulkan_compute)
    {
        ReadSprivMapBinaryFile(model_name);//create_pipeline() spriv˳��
    }
    ReadBinaryFile(model_name, use_vulkan_compute);//cpu����˳��

    int file_bin =0;
    char tbinpath[256];
    sprintf(tbinpath, MODEL_DIR "%s.bin", model_name);
    FILE* tfp = fopen(tbinpath, "rb");
    if (tfp)
    {
        file_bin =1;
    }

    //ARM
    // 0 1: run
    // 1 1: test
    // 0 0: unable pack
    ARM_W_TEST = 0;
    USE_PACK_ARM = 1 - device_cpus;
    // 1: select kernal
    USE_KERNAL_ARM = 1;
    // 1: pipeline
    int USE_PIPLIEN = 1;

    if(USE_PACK_ARM){
        if(ARM_W_TEST)
        {
            arm_weight_file_init(model_name);
        }
        else{
            arm_weight_file_read_init(model_name);
            ReadarmWeightDataReaderFile(model_name);
        }
    }
    if(USE_PIPLIEN && (ARM_W_TEST == 0)){
        benchmark_new(model_name, in, opt);
    }
    else{
        benchmark(model_name, in, opt);
    }

    if(USE_PACK_ARM){
        if(ARM_W_TEST)
        {
            fclose(arm_weight_file);
            WritearmWeightDataReaderFile(model_name);
            ReadarmWeightDataReaderFile(model_name);
        }
        else{
            fclose(arm_weight_file_read);
            for(int i=0; i < 8; i++){
                fclose(arm_weight_file_reads[i]);
            }
        }
    }
    WriteOutputFile(model_name);

#if NCNN_VULKAN
    delete g_blob_vkallocator;
    delete g_staging_vkallocator;
#endif // NCNN_VULKAN

    return 0;
}
