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
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <fstream>
#include <functional>

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
#ifdef __EMSCRIPTEN__
#define MODEL_DIR "/working/"
#else
#define MODEL_DIR ""
#endif
//
//export GOMP_CPU_AFFINITY="7 6 5 4";

char globalbinpath[256];
char gloableparampath[256];
//size_t dr_cpu=0;//5;
//size_t pipe_cpu=1;//6;
//size_t infer_cpu=7;//0;

double start_t = 0;

//ncnn::Net net;


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

/////////////////////////////////////////////////////////////////////////////
std::vector<double> rc_times(1000, -1);
//DataReaderFromEmpty drp;
//ncnn::ModelBinFromDataReader mb(drp);

//#define CPU_NUMS 4
//int CPU_LITTLE_s[CPU_NUMS] = {0, 1, 2, 3};
int CPU_NUMS;// = 4;
std::vector<int> CPU_LITTLE_s;// = {0, 1, 2, 3};
int CPU_BIG_;// = 7;
//std::vector<int> cpuLittle_Vectors[CPU_NUMS];
std::vector<std::vector<int>> cpuLittle_Vectors;//(CPU_NUMS);
std::vector<int> cpuBig_Vector;
//pthread_cond_t param_cond_cpus[CPU_NUMS];
std::vector<pthread_cond_t> param_cond_cpus;//(CPU_NUMS);
//double last_cpu_times[CPU_NUMS];
std::vector<double> last_cpu_times;//(CPU_NUMS);
//double cpu_times[CPU_NUMS];
std::vector<double> cpu_times; //(CPU_NUMS);

void setDeviceConfig(int device_cpus){
    if(device_cpus == 0){
        CPU_NUMS= 4;
        CPU_LITTLE_s.resize(CPU_NUMS);
        CPU_LITTLE_s = {0, 1, 2, 3};
        CPU_BIG_ = 7;
        cpuLittle_Vectors.resize(CPU_NUMS);
        param_cond_cpus.resize(CPU_NUMS);
        last_cpu_times.resize(CPU_NUMS);
        cpu_times.resize(CPU_NUMS);
    }
    else if(device_cpus == 1){
        CPU_NUMS= 2;
        CPU_LITTLE_s.resize(CPU_NUMS);
        CPU_LITTLE_s = {1, 2};
        CPU_BIG_ = 0;
        cpuLittle_Vectors.resize(CPU_NUMS);
        param_cond_cpus.resize(CPU_NUMS);
        last_cpu_times.resize(CPU_NUMS);
        cpu_times.resize(CPU_NUMS);
    }
    else{
        CPU_NUMS= 4;
        CPU_LITTLE_s.resize(CPU_NUMS);
        CPU_LITTLE_s = {0, 1, 2, 3};
        CPU_BIG_ = 7;
        cpuLittle_Vectors.resize(CPU_NUMS);
        param_cond_cpus.resize(CPU_NUMS);
        last_cpu_times.resize(CPU_NUMS);
        cpu_times.resize(CPU_NUMS);
    }
}

////////////////////////files//////////////////////////////////

void WriteBinaryFile( const char* comment, bool use_vulkan_compute)
{
    char path[256];
    if(use_vulkan_compute){
        sprintf(path, MODEL_DIR "%s.vk.dat", comment);
    }
    else
    {
        sprintf(path, MODEL_DIR "%s.dat", comment);
    }
    std::ofstream osData(path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

    for (int i=0; i<CPU_NUMS; i++)
    {
        int n = CPU_NUMS - i - 1;
        int cpuListSize = cpuLittle_Vectors[n].size();
        osData.write(reinterpret_cast<char *>(&cpuListSize), sizeof(cpuListSize));
        osData.write(reinterpret_cast<char *>(cpuLittle_Vectors[n].data()), cpuListSize*sizeof(cpuLittle_Vectors[n].front()) );

    }
    int cpu7ListSize = cpuBig_Vector.size();
    osData.write(reinterpret_cast<char *>(&cpu7ListSize), sizeof(cpu7ListSize));
    osData.write(reinterpret_cast<char *>(cpuBig_Vector.data()), cpu7ListSize*sizeof(cpuBig_Vector.front()) );
    osData.close();
    std::cout<<"[WRITE]["<<path<<"]cpuLittle_Vectors"<<std::endl;
}
void WriteDataReaderFile( const char* comment)
{
    char path[256];
    sprintf(path, MODEL_DIR "%s.br.dat", comment);
    std::ofstream osData(path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

    int DRListSize = DR_file_Vectors.size();
    std::cout<<"[WRITE]["<<path<<"]DR_file_Vectors(len:"<<DRListSize<<"):{";
    for(int i : DR_file_Vectors){
        printf("%d,", i);
    }
    std::cout<<"}"<<std::endl;

    osData.write(reinterpret_cast<char *>(&DRListSize), sizeof(DRListSize));
    osData.write(reinterpret_cast<char *>(DR_file_Vectors.data()), DRListSize*sizeof(DR_file_Vectors.front()) );

    osData.close();
}
void WriteSprivMapBinaryFile( const char* comment)
{
    char path[256];
    sprintf(path, MODEL_DIR "%s.m.dat", comment);
    std::ofstream osData(path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

    int ssSize = ncnn::spriv_map.size();
    osData.write(reinterpret_cast<char *>(&ssSize), sizeof(ssSize));
    std::cout<<"[WRITE]["<<path<<"]spriv_map(map_len:"<<ssSize<<"):{";
    for(auto &it:ncnn::spriv_map)
    {
        auto k = it.first;
        osData.write(reinterpret_cast<char *>(&k), sizeof(k));
        auto s = it.second;
        int sSize = s.size();
        osData.write(reinterpret_cast<char *>(&sSize), sizeof(sSize));
        osData.write(reinterpret_cast<char *>(s.data()), sSize*sizeof(s.front()) );
        std::cout<<it.first<<", ";
    }
    std::cout<<"}"<<std::endl;
    osData.close();
}
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


////////////////////////benchmark//////////////////////////////////
void benchmark(const char* comment, const ncnn::Mat& _in, const ncnn::Option& opt)
{
    std::cout<<"[BENCHMARK][START]"<<std::endl;
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
    if(USE_PACK_ARM){
        arm_weight_file_seek_Vectors.resize(net.layers().size());
    }

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
        double time = end - start;
        if (i==0){
            first_time = time;
        }
        //        fprintf(stderr, "%f\n", time);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i = 0; i < g_loop_count; i++)
    {
        double start = ncnn::get_current_time();

        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input(input_names[0], in);
            ex.extract(output_names[0], out);
            float* ptr = (float*)out.data;
            //            printf("out %f\n", *ptr);
        }

        double end = ncnn::get_current_time();

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
        //        fprintf(stderr, "%f\n", time);
    }

    time_avg /= g_loop_count;

    //    fprintf(stderr, "%20s  load param = %7.2f  load model = %7.2f  create pipeline = %7.2f  first = %7.2f  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, load_param_time, load_model_time, load_pipe_time, first_time, time_min, time_max, time_avg);
    printf("_____%15s  load param = %7.2f  load model = %7.2f  create pipeline = %7.2f  first = %7.2f  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, load_param_time, load_model_time, load_pipe_time, first_time, time_min, time_max, time_avg);
    std::cout<<"[BENCHMARK][FINISH]"<<std::endl;
}

////////////////////////benchmark_biglittle//////////////////////////////////
double largecore_rc_time=0;
void inference(const ncnn::Net& net, const ncnn::Mat& in, bool print=true){
    for_cal_time = 0;
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
        std::cout<<"_____    out[0] = "<<*ptr<<std::endl;
        std::cout<<"_____    for_skp_time="<<for_skp_time<<" for_cal_time="<<for_cal_time<<std::endl;
        //        fprintf(stderr, "infer time =  %7.2f\n", infer_time);
        std::cout<<"_____    infer time = "<<infer_time<<std::endl;
    }

    //    pthread_exit(&t_infer);

}

void * thread_cpuBig(void *args){
#ifdef __EMSCRIPTEN__
#define MODEL_DIR "/working/"
#else
#define MODEL_DIR ""
#endif
    param_finish_1 = 0;
#ifdef __CPU_MASK__
    cpu_set_t mask;                                      //CPU核的集合
    CPU_ZERO(&mask);                                     //置空
    CPU_SET(7, &mask);                                   //设置亲和力值
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) //设置线程CPU亲和力
    {
        printf("warning: could not set CPU affinity, continuing...\n");
    }
#endif
    finish_set_init();

    auto *net=(ncnn::Net*)args;
    ncnn::current_layer_idx_f2p = 0;
    ncnn::current_layer_idx_p2i = 0;
    infer_start = 0;
    pipe_start = 0;
    for_cal_time = 0;
    for_skp_time = 0;
    read_syn = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, {}};
    create_syn = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, {}};
    infer_syn = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, {}};




    pthread_mutex_lock(&param_lock);
    param_finish_1 = 100;
    pthread_mutex_unlock(&param_lock);
    for ( int i=0; i<CPU_NUMS; i++)
    {
        pthread_cond_signal(&param_cond_cpus[i]);
    }
    pthread_cond_signal(&param_cond_cpu3);
    pthread_cond_signal(&param_cond_cpu2);
    pthread_cond_signal(&param_cond_cpu1);
    pthread_cond_signal(&param_cond_cpu0);
    pthread_cond_signal(&param_cond_cpu7);
    pthread_cond_signal(&param_cond_cpu6);
    pthread_cond_signal(&param_cond_cpu5);
    pthread_cond_signal(&param_cond_cpu4);


//    printf("param finish_____________\n");
    printf("_____    Thread_exec_(cpu.large)     tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
//    std::cout<<"_____    Thread_exec_(cpularge)     tid="<<pthread_self()<<",cpu="<<sched_getcpu()<<std::endl;
    double start_time = ncnn::get_current_time();
    /*read*/
//    FILE* fp = fopen(globalbinpath, "rb");
//    printf("_____    load file %s\n", globalbinpath);
//    if (!fp)
//    {
////        NCNN_LOGE("fopen %s failed", globalbinpath);
//        printf("fopen %s failed", globalbinpath);
//        //        return 0;
//    }
//    else
//    {
//        ncnn::DataReaderFromStdio dr(fp);
//        if (net->layers().empty())
//        {
////            NCNN_LOGE("network graph not ready");
//            printf("network graph not ready");
//            //        return 0;
//        }
//        ncnn::ModelBinFromDataReader mb1(dr);
//
//        mb = mb1;
//    }
    DataReaderFromEmpty dr;
    ncnn::ModelBinFromDataReader mb(dr);//get_dr(fp, net));

    /*create*/
    double s0 = ncnn::get_current_time();
    net->load_model_layer(mb, 0);
    if(USE_PACK_ARM)
    {
        fseek(arm_weight_file_reads[sched_getcpu()], arm_weight_file_seek_Vectors[0], SEEK_SET);
    }
    net->load_pipe_layer(0);
    net->upload_model_layer(0);
    double e0 = ncnn::get_current_time();
    rc_times[0] = e0-s0;

    double s1 = ncnn::get_current_time();
    net->load_model_layer(mb, 1);
    syn_act(read_syn, 2);
    if(USE_PACK_ARM)
    {
        fseek(arm_weight_file_reads[sched_getcpu()], arm_weight_file_seek_Vectors[0], SEEK_SET);
    }
    net->load_pipe_layer(1);
    net->upload_model_layer(1);
    double e1 = ncnn::get_current_time();
    rc_times[1] = e1-s1;
    syn_act(infer_syn, 1);

    //    int list_len = sizeof(cpu7_list)/ sizeof(cpu7_list[0]);
    //    double cal_time= 0;
    //    for(int ii=0; ii<list_len; ii++){
    //        int i = cpu7_list[ii];
    for(int i: cpuBig_Vector){
        //        double s = ncnn::get_current_time();
        net->load_model_layer(mb, i);
        if(USE_PACK_ARM)
        {
            fseek(arm_weight_file_reads[sched_getcpu()], arm_weight_file_seek_Vectors[i-1], SEEK_SET);
        }
        net->load_pipe_layer(i);
        net->upload_model_layer(i);
        //        double e = ncnn::get_current_time();
        //        rc_times[i] = e-s;
        syn_act(infer_syn, i);
    }

    double start_time_inf = ncnn::get_current_time();
    inference(*net, ncnn::Mat(224, 224, 3), true);
    double end_time = ncnn::get_current_time();
    largecore_rc_time = start_time_inf-s0;
    printf("_____    Thread_exec_(cpu.large)[FINISH]     rc=%f infer=%f time = %f\n", start_time_inf-s0, end_time - start_time_inf, end_time - start_time);
//    if (fp)
//    {
//        fclose(fp);
//    }

    param_finish_1 = 0;
    return 0;
}

void * thread_cpuLittle(void *args){

    pthread_mutex_lock(&param_lock);
    while (param_finish_1 == 0 ){
        pthread_cond_wait(&param_cond_cpu3, &param_lock);
    }
    pthread_mutex_unlock(&param_lock);

#ifdef __CPU_MASK__
    cpu_set_t mask;  //CPU核的集合
    CPU_ZERO(&mask);    //置空
    CPU_SET(3,&mask);   //设置亲和力值
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//设置线程CPU亲和力
    {
        printf("warning: could not set CPU affinity, continuing...\n");
    }
#endif
    printf("_____    Thread_load&pipe_(cpu.little)    tid=%ld,cpu=%d\n",  pthread_self(), sched_getcpu());
    double start_time = ncnn::get_current_time();

    DataReaderFromEmpty dr;
    ncnn::ModelBinFromDataReader mb(dr);//get_dr(fp, net));

    auto *net=(ncnn::Net*)args;
    int layer_count = (int)net->layers().size();
    int start_i=2;
    for(int i=start_i; i<layer_count;){
        double s = ncnn::get_current_time();
#ifdef __CPU_MASK__
        if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//设置线程CPU亲和力
        {
            printf("warning: could not set CPU affinity, continuing...\n");
        }
#endif
        net->load_model_layer(mb, i);
        if(USE_PACK_ARM)
        {
            fseek(arm_weight_file_reads[sched_getcpu()], arm_weight_file_seek_Vectors[i-1], SEEK_SET);
        }
        net->load_pipe_layer(i);
        net->upload_model_layer(i);
        double e = ncnn::get_current_time();
        rc_times[i] = e-s;

        if(timeshow)
        {
            printf("%d,%f\n", i, e - s);
        }

        syn_act(infer_syn, i);

        pthread_mutex_lock(&next_layer_lock);
        i = select_next_layer();
        pthread_mutex_unlock(&next_layer_lock);
    }
    double end_time = ncnn::get_current_time();
    printf("_____    Thread_load&pipe_(cpu.little)[FINISH]    time = %ff\n", end_time - start_time);
    return 0;
}

void benchmark_biglittle(const char* comment, const ncnn::Mat& _in, const ncnn::Option& opt)
{
    std::cout<<"=============benchmark_biglittle[START]=========="<<std::endl;
    //    rc_times.clear();
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
    layer_next=2;
    timeshow=0;
    char parampath[256];
    sprintf(parampath, MODEL_DIR "%s.param", comment);
    sprintf(gloableparampath, MODEL_DIR "%s.param", comment);
    char binpath[256];
    sprintf(binpath, MODEL_DIR "%s.bin", comment);
    sprintf(globalbinpath, MODEL_DIR "%s.bin", comment);


    net.load_param(parampath);
    rc_times = std::vector<double>(net.layers().size(), -1);

    std::thread t_big(thread_cpuBig, (void*)&net);
    std::thread t_little(thread_cpuLittle, (void*)&net);
    t_big.join();
    t_little.join();
    std::cout<<"=============benchmark_biglittle[FINISH] rc_times_len="<<rc_times.size()<<"==========\n";
}

////////////////////////benchmark_new//////////////////////////////////
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

//    pthread_mutex_lock(&param_lock);
//    while (param_finish_1 == 0 ){
//        //        printf("=================\n");
//        pthread_cond_wait(&param_cond_cpus[net_cid_->cpu_id], &param_lock);
//        //        sleep(0);
//    }
//    pthread_mutex_unlock(&param_lock);

#ifdef __CPU_MASK__
    cpu_set_t mask;  //CPU核的集合
    CPU_ZERO(&mask);    //置空
    CPU_SET(net_cid_->cpu_set,&mask);   //设置亲和力值
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//设置线程CPU亲和力
    {
        printf("warning: could not set CPU affinity, continuing...\n");
    }
#endif
//    printf("__________t%d_cpu%d    tid=%ld,cpu=%d\n",net_cid_->cpu_id, net_cid_->cpu_set, pthread_self(), sched_getcpu());
    printf("_____    Thread_load&pipe_(cpu%d)([%d])    tid=%ld,cpu=%d\n",net_cid_->cpu_set,net_cid_->cpu_id,  pthread_self(), sched_getcpu());
//    std::cout<<"_____    Thread_load&pipe_(cpu"<<net_cid_->cpu_set<<")(["<<sched_getcpu()<<"])     tid="<<pthread_self()<<",cpu="<<sched_getcpu()<<std::endl;
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
        net_cid_->net->upload_model_layer(i);
        double e = ncnn::get_current_time();
        rc_times[i] = e-s;
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
    cpu_times[net_cid_->cpu_id] = end_time - start_time;
//    printf("_____    cpu_times[%d] = %f \n", net_cid_->cpu_id, end_time - start_time);
    printf("_____    Thread_load&pipe_(cpu%d)([%d])[FINISH]        time = %f %f\n", net_cid_->cpu_set,sched_getcpu(), end_time - start_time, cal_time);
//    std::cout<<"_____    Thread_load&pipe_(cpu"<<net_cid_->cpu_set<<")(["<<sched_getcpu()<<"])[FINISH]     time="<<end_time - start_time<<",cal time="<<cal_time<<std::endl;
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
//    std::cout<<"_____    Thread_load&pipe_(cpu"<<net_cid_->cpu_set<<")(["<<sched_getcpu()<<"])     tid="<<pthread_self()<<",cpu="<<sched_getcpu()<<std::endl;
    printf("_____    Thread_load&pipe_(cpu%d)([%d])    tid=%ld,cpu=%d\n",net_cid_->cpu_set,net_cid_->cpu_id,  pthread_self(), sched_getcpu());
    double start_time = ncnn::get_current_time();


    //        auto net=net_cid_->net;
    int list_len = cpuLittle_Vectors[net_cid_->cpu_id].size();
    double cal_time= 0;
    for(int ii=0; ii<list_len; ii++){
        int i = cpuLittle_Vectors[net_cid_->cpu_id][ii];
        double s = ncnn::get_current_time();

#ifdef __CPU_MASK__
        if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//�����߳�CPU�׺���
        {
            printf("warning: could not set CPU affinity, continuing...\n");
        }
#endif
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
        rc_times[i] = e-s;
//        double e = ncnn::get_current_time();

        cal_time+= e-s;
        if(timeshow)
        {
            printf("_____    load&pipe %d,%f\n", i, e - s);
        }
        //        net->load_model_layer(mb, i);
        //        net->load_pipe_layer(i);
        syn_act(infer_syn, i);
    }
    double end_time = ncnn::get_current_time();
    cpu_times[net_cid_->cpu_id] = end_time - start_time;
//    printf("_____    cpu_times[%d] = %f \n", net_cid_->cpu_id, end_time - start_time);
    printf("_____    Thread_load&pipe_(cpu%d)([%d])[FINISH]        time = %f %f\n", net_cid_->cpu_set,sched_getcpu(), end_time - start_time, cal_time);
//    std::cout<<"_____    Thread_load&pipe_(cpu"<<net_cid_->cpu_set<<")(["<<sched_getcpu()<<"])[FINISH]     time="<<end_time - start_time<<",cal time="<<cal_time<<std::endl;

    if (fp_)
    {
        fclose(fp_);
    }
    return 0;
}

void cold_boot_empty(ncnn::Net& net, const ncnn::Mat& in)
{
//    std::cout<<"_____    Thread_exec_(cpu4567)     tid="<<pthread_self()<<",cpu="<<sched_getcpu()<<std::endl;
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

    double s0 = ncnn::get_current_time();
    net.load_model_layer(mb, 0);
    net.load_pipe_layer(0);
    double e0 = ncnn::get_current_time();
    rc_times[0] = e0-s0;

    double s1 = ncnn::get_current_time();
    net.load_model_layer(mb, 1);
    syn_act(read_syn, 2);
    net.load_pipe_layer(1);
    syn_act(infer_syn, 1);
    double e1 = ncnn::get_current_time();
    rc_times[1] = e1-s1;


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

    for(int i = 0; i<CPU_NUMS; i++){
        th[i].join();
    }

    double start_time_inf = ncnn::get_current_time();
    inference(net, in, true);
    double end_time = ncnn::get_current_time();
    largecore_rc_time = start_time_inf-s0;
//    std::cout<<"_____    Thread_exec_(cpu4567)[FINISH]     rc="<<start_time_inf-s<<",infer="<<end_time - start_time_inf<<std::endl;
    printf("_____    Thread_exec_(cpu4567)[FINISH]     rc=%f infer=%f\n", start_time_inf-s, end_time - start_time_inf);
    //    t_0.join();
    //    t_1.join();
//    for(int i = 0; i<CPU_NUMS; i++){
//        th[i].join();
//    }
}

void cold_boot_file(ncnn::Net &net, FILE* fp, const ncnn::Mat& in){
//    std::cout<<"_____    Thread_exec_(cpu4567)     tid="<<pthread_self()<<",cpu="<<sched_getcpu()<<std::endl;
    printf("_____    Thread_exec_(cpu4567)     tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    if(USE_PACK_ARM)
    {
        for(int i=0; i<8; i++){
            fseek(arm_weight_file_reads[sched_getcpu()], 0, SEEK_SET);
        }
    }
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

    double s0 = ncnn::get_current_time();
    fseek(fp, DR_file_Vectors[0], SEEK_SET);
    net.load_model_layer(mb, 0);
    net.load_pipe_layer(0);
    //#if NCNN_VULKAN
    //    fseek(vk_weight_file_read,  vk_weight_file_seek_Vectors[0], SEEK_SET);
    //#endif
    net.upload_model_layer(0);
    double e0 = ncnn::get_current_time();
    rc_times[0] = e0-s0;

    double s1 = ncnn::get_current_time();
    fseek(fp, DR_file_Vectors[1], SEEK_SET);
    net.load_model_layer(mb, 1);
    syn_act(read_syn, 2);
    net.load_pipe_layer(1);
    //#if NCNN_VULKAN
    //    fseek(vk_weight_file_read,  vk_weight_file_seek_Vectors[1], SEEK_SET);
    //#endif
    net.upload_model_layer(1);
    syn_act(infer_syn, 1);
    double e1 = ncnn::get_current_time();
    rc_times[1] = e1-s1;

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
    largecore_rc_time = start_time_inf-s0;
//    std::cout<<"_____    Thread_exec_(cpu4567)[FINISH]     rc="<<start_time_inf-s<<",infer="<<end_time - start_time_inf<<std::endl;
    printf("_____    Thread_exec_(cpu4567)[FINISH]     rc=%f infer=%f\n", start_time_inf-s, end_time - start_time_inf);
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

int warmUp = 0;
void benchmark_new(const char* comment, const ncnn::Mat& _in, const ncnn::Option& opt)
{
    std::cout<<"=============benchmark_new[START]=========="<<std::endl;
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
    rc_times = std::vector<double>(net.layers().size(), -1);
    //    printf("ssssssssssssssss  %d\n", net.layers().size());



//    printf("start %f\n", ncnn::get_current_time()-start_t);
//    printf("\n==s====================\n");
//    printf("\n=========s=============\n");
    double s_time = ncnn::get_current_time();
//    int warmUp = 0;
    if(warmUp) //Warm up SoC to max the SoC CPU's freq.
    {
        ncnn::Net net_in;
        net_in.opt = net.opt;
//        char tparampath[256];
//        sprintf(tparampath, MODEL_DIR "mobilenet_v3.param"); //GoogleNet  //AlexNet
//        char tbinpath[256];
//        sprintf(tbinpath, MODEL_DIR "mobilenet_v3.bin");
        net_in.load_param(gloableparampath);
        int open = net_in.load_model_dr(globalbinpath);
        if (open < 0)
        {
//            printf("load file files\n");
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
//    clear_times_save();
//    resize_times_save(net.layers().size());

//    printf("param finish_____________\n");
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
//    printf("_____    Thread_exec_(cpu4567)     tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    double start_time = ncnn::get_current_time();
    save_start_time = ncnn::get_current_time();
    //read
//
    FILE* fp = fopen(globalbinpath, "rb");
    if (!fp)
    {
//        NCNN_LOGE("_____    fopen %s failed", globalbinpath);
        std::cout<<"_____    fopen "<<globalbinpath<<" failed"<<std::endl;
        cold_boot_empty(net, in);
    }
    else
    {
        std::cout<<"_____    load file "<<globalbinpath<<std::endl;
        if (net.layers().empty())
        {
//            NCNN_LOGE("_____    network graph not ready");
            printf("_____    network graph not ready");
        }
        cold_boot_file(net, fp, in);
    }

//    printf("_____________________________________________________________________________total time %f\n", ncnn::get_current_time()-start_time);
//
//    printf("_____________________________________________________________________________total time + warmup %f\n", ncnn::get_current_time()-s_time);
//    printf("_____________________________________________________________________________real total time %f\n", ncnn::get_current_time()-start_t);
//    printf("==============================================\n");
//    printf("=============benchmark_new[FINISH]___total_time=%f____total_time_+_warmup=%f___real_total_time=%f ==========\n",  ncnn::get_current_time()-start_time, ncnn::get_current_time()-s_time, ncnn::get_current_time()-start_t);
//    printf("=============benchmark_new[FINISH] ==========\n");
    std::cout<<"=============benchmark_new[FINISH]___total_time="<<ncnn::get_current_time()-start_time<<"____total_time_+_warmup="<<ncnn::get_current_time()-s_time<<"___real_total_time="<<ncnn::get_current_time()-start_t<<" =========="<<std::endl;

}

////////////////////////get_queue//////////////////////////////////

template<typename T>
T SumVector(std::vector<T>& vec)
{
    T res = 0;
    for (size_t i=0; i<vec.size(); i++)
    {
        res += vec[i];
    }
    return res;
}

double loop_sum=0, last_loop_sum=0,last_infer_time =0, last_largecore_rc_time=0;
double large_rc_need=0;
double eta=0.25;
int finish_f=0;
void get_largecore_queue(int cnt=0){

    if(cnt==0){
        return;
    }

    std::cout<<"=============get_largecore_queue[START]=========="<<std::endl;
    printf("_____    sum=%f rc_time=%f last_infer_time=%f,  last_CPU_time =[", last_loop_sum+0, last_largecore_rc_time, last_infer_time);
//    for (double last_cpu_time:last_cpu_times)
//    {
//        printf("%f, ", last_cpu_time);
//    }
//    printf("]\n");
    //    printf("%f %f %f %F\n", last_cpu3_time, last_cpu2_time, last_cpu1_time, last_cpu0_time);

    //    double cpu_times[]={last_cpu0_time, last_cpu1_time, last_cpu2_time, last_cpu3_time};
//    double max_cpu_time = *std::max_element(last_cpu_times,last_cpu_times+CPU_NUMS)*1.75;
    double max_cpu_time = 0;
    for (double last_cpu_time : last_cpu_times)
    {
        printf("%f, ", last_cpu_time);
        max_cpu_time += last_cpu_time/CPU_NUMS;
    }
    printf("]\n");
//    max_cpu_time = max_cpu_time*1.5;
    double diff_time = max_cpu_time -(last_largecore_rc_time + last_infer_time);


//    printf("_____    diff %f t%f m%f\n", diff_time, last_largecore_rc_time + last_infer_time, max_cpu_time);
    if(diff_time<=0&& cpuBig_Vector.size()==0){
        finish_f=1;
        return;
    }
    if(diff_time<0)
        large_rc_need =diff_time*0.5*eta + large_rc_need*(1-0.5*eta);
    else
        large_rc_need =diff_time*eta + large_rc_need*(1-eta);
    printf("_____    time_cpu7:%f max_time_cpu1-4:%f need_time_cpu7:%f diff:%f \n",last_largecore_rc_time + last_infer_time, max_cpu_time, large_rc_need, diff_time);

    std::vector<int> idx_vector;
    for(int i=2; i<rc_times.size(); i++) {
        idx_vector.push_back(i);
    }

    int len = idx_vector.size();
    int i, j; double temp;
    for (i = 0; i < len - 1; i++){
        for (j = 0; j < len - 1 - i; j++)
        {
            if (rc_times[idx_vector[j]] < rc_times[idx_vector[j + 1]])
            {
                temp = idx_vector[j];
                idx_vector[j] = idx_vector[j + 1];
                idx_vector[j + 1] = temp;
            }
        }
    }
    //    for(int i: idx_vector){
    //        printf("%d,", i);
    //    }
    //    printf("\n");

    std::vector<int> cpu7_Vector_;
    double ssum=0;
    printf("_____    cpu7_rc_times:[");
    for(int idx :idx_vector){
        printf("%d:%f, ", idx, rc_times[idx]);
        if(ssum + rc_times[idx]<=large_rc_need)
        {
            ssum+=rc_times[idx];
            cpu7_Vector_.push_back(idx);
        }
    }
    printf("]\n");

    std::sort(cpu7_Vector_.begin(), cpu7_Vector_.end());
    cpuBig_Vector = cpu7_Vector_;
    printf("_____    cpuBig_Vector:[");//, large_rc_need, diff_time);
    for (auto v : cpuBig_Vector)
    {
        printf("%d,", v);
    }
    printf("]\n");

    //    if(diff_time>=0)
    //    {
    ////        double large_rc_need = diff_time;
    ////        cpuBig_Vector+
    //    }
    //    else{
    ////        cpuBig_Vector-
    //    }
    //    cpuBig_Vector.push_back(2);
    //    cpuBig_Vector.push_back(5);
    //    cpuBig_Vector.push_back(110);
    std::cout<<"=============get_largecore_queue[FINISH]=========="<<std::endl;
    return;
}
void get_queue_init(){
    std::cout<<"=============get_queue_init[START]=========="<<std::endl;
    std::vector<int> qs[CPU_NUMS];
    std::vector<double> ts[CPU_NUMS];
    double sums[CPU_NUMS];
    for(int i=2; i<rc_times.size(); i++){
        if(std::count(cpuBig_Vector.begin(), cpuBig_Vector.end(), i))// i not in cup7
            continue;

        for (int j=0; j<CPU_NUMS; j++)
        {
            sums[j] = SumVector(ts[j]);
        }

        int minPosition = std::min_element(sums,sums+CPU_NUMS) - sums;
        qs[minPosition].push_back(i);
        ts[minPosition].push_back(rc_times[i]);

    }

    printf("_____    exec_time SUM:  ");
    for (int j=0; j<CPU_NUMS; j++)
    {
        printf("%f  ", sums[j]);
        cpuLittle_Vectors[j]= qs[j];
    }
    printf("\n");
    std::cout<<"=============get_queue_init[FINISH]=========="<<std::endl;

    return;
}
void get_queue()
{
    std::cout<<"=============*get_queue[START]=========="<<std::endl;
    std::vector<double> ts[CPU_NUMS];
    double sums[CPU_NUMS];
    //    std::vector<double> t1, t2, t3, t0;
    for (int j=0; j<CPU_NUMS; j++)
    {
        for (int i : cpuLittle_Vectors[j])
        {
            ts[j].push_back(rc_times[i]);
        }
    }

    for (int j=0; j<CPU_NUMS; j++)
    {
        sums[j] = SumVector(ts[j]);
    }

    printf("_____    SUM  ");
    for (int j=0; j<CPU_NUMS; j++)
    {
        printf("%f  ", sums[j]);
    }
    printf("\n");
    //    double sum[]={sum0, sum1, sum2, sum3};
    int minPosition = std::min_element(sums,sums+CPU_NUMS) - sums;
    int maxPosition = std::max_element(sums,sums+CPU_NUMS) - sums;
    std::vector<int> max_vector = cpuLittle_Vectors[maxPosition];
    std::vector<int> min_vector = cpuLittle_Vectors[minPosition];

    int len = max_vector.size();
    int i, j; double temp;
    for (i = 0; i < len - 1; i++)
        for (j = 0; j < len - 1 - i; j++)
            if (rc_times[max_vector[j]] < rc_times[max_vector[j + 1]]) {
                temp = max_vector[j];
                max_vector[j] = max_vector[j + 1];
                max_vector[j + 1] = temp;
            }
    double diff = sums[maxPosition] - loop_sum/4.0;

    std::vector<int> max_vector_;
    for(int i=0; i<max_vector.size(); i++){
        if(rc_times[max_vector[i]]<=diff){
            for(int k = i ; k< max_vector.size(); k++){
                //                max_vector_.clear();
                if(max_vector[k]!= max_vector[i]){
                    max_vector_.push_back(max_vector[k]);
                }
                else{
                    min_vector.push_back(max_vector[k]);
                }
            }
            break;
        }
        else
            max_vector_.push_back(max_vector[i]);
    }
    std::sort(min_vector.begin(), min_vector.end());

    cpuLittle_Vectors[maxPosition] = max_vector_;
    cpuLittle_Vectors[minPosition] = min_vector;
    for (int j=0; j<CPU_NUMS; j++)
    {
        std::sort(cpuLittle_Vectors[j].begin(), cpuLittle_Vectors[j].end());
    }


    std::vector<double> t_s[CPU_NUMS];
    for (int j=0; j<CPU_NUMS; j++)
    {
        for (int i : cpuLittle_Vectors[j])
        {
            t_s[j].push_back(rc_times[i]);
        }
    }

    for (int j=0; j<CPU_NUMS; j++)
    {
        sums[j] = SumVector(t_s[j]);
    }
    for (int j=0; j<CPU_NUMS; j++)
    {
        printf("_____    int cpu%d_list[]={", j);
        for (int i = 0; i < cpuLittle_Vectors[j].size(); ++i)
        {
            printf("%d,", cpuLittle_Vectors[j][i]);
        }
        printf("};\n");
    }
    printf("_____    int cpu7_list[]={");
    for (int i = 0; i < cpuBig_Vector.size(); ++i)
    {
        printf("%d,", cpuBig_Vector[i]);
    }
    printf("};\n");


    printf("_____    exec_time SUM  ");
    for (int j=0; j<CPU_NUMS; j++)
    {
        printf("%f  ", sums[j]);
    }
    printf("\n");
    //    printf("SUM  %f, %f, %f, %f\n", sum0, sum1, sum2, sum3);
    std::cout<<"=============*get_queue[FINISH]=========="<<std::endl;
}
double sumRCtime(int printF)
{
    double sum_time = 0;
    if(printF)
        std::cout<<"_____    load&pipe's latency: rc_times:[";
    for (int i = 0; i < rc_times.size(); i++)
    {
        if(printF)
            std::cout<< rc_times[i]<<", ";
        sum_time += rc_times[i];
    }
    if(printF)
        std::cout<<"]"<<std::endl;
    return sum_time;
}

int main(int argc, char** argv)
{
    TEST_=1;

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
    if (use_vulkan_compute)
    {
        g_warmup_loop_count = 10;

        g_vkdev = ncnn::get_gpu_device(gpu_device);

        g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
        g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
    }
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

//        char model_name[] = "alexnet";
//        ncnn::Mat in = ncnn::Mat(227, 227, 3);
//
//        char model_name[] = "googlenet";
//        ncnn::Mat in = ncnn::Mat(224, 224, 3);
//
//        char model_name[] = "MobileNet";
//        ncnn::Mat in = ncnn::Mat(224, 224, 3);
//
//        char model_name[] = "MobileNetV2";
//        ncnn::Mat in = ncnn::Mat(224, 224, 3);
//
//        char model_name[] = "resnet18";
//        ncnn::Mat in = ncnn::Mat(224, 224, 3);
//
//        char model_name[] = "shufflenet";
//        ncnn::Mat in = ncnn::Mat(224, 224, 3);
//
//        char model_name[] = "efficientnet_b0";
//        ncnn::Mat in = ncnn::Mat(224, 224, 3);
//
//        char model_name[] = "resnet50";
//        ncnn::Mat in = ncnn::Mat(224, 224, 3);
//
//        char model_name[] = "squeezenet";
//        ncnn::Mat in = ncnn::Mat(227, 227, 3);
//
//        char model_name[] = "shufflenet_v2";
//        ncnn::Mat in = ncnn::Mat(224, 224, 3);
//
//        char model_name[] = "yolov4-tiny";
//        ncnn::Mat in = ncnn::Mat(416, 416, 3);
//
//        char model_name[] = "mobilenetv2_yolov3";
//        ncnn::Mat in = ncnn::Mat(352, 352, 3);
//
//        char model_name[] = "mobilenet_yolo";
//        ncnn::Mat in = ncnn::Mat(416, 416, 3);

    ncnn::Mat in = ncnn::Mat(224, 224, 3);

    TEST_ = 1;
    // 1: select kernal
    USE_KERNAL_ARM = 1;

    int outerLoop = 5;
    int interLoop = 10;//10

    double sss = ncnn::get_current_time();
    int file_bin =0;
    char tbinpath[256];
    sprintf(tbinpath, MODEL_DIR "%s.bin", model_name);
    FILE* tfp = fopen(tbinpath, "rb");
    if (tfp)
    {
        file_bin =1;
    }
    std::cout<<"==========SAVE-TransWeight-TransWeightIdx-OriginWeightIdx[START]=========="<<std::endl;
    ARM_W_TEST = 1;
    USE_PACK_ARM = 1 - device_cpus;
    if(USE_PACK_ARM){
        if(ARM_W_TEST)
        {
            arm_weight_file_init(model_name); //初始化文件“%NAME%.arm.bin"，用于存储weight_transform后的weight。
        }
        else{
            arm_weight_file_read_init(model_name);//打开文件“%NAME%.arm.bin"
            ReadarmWeightDataReaderFile(model_name);//读取文件“%NAME%.arm.br.dat"得到weight_transform后的weight的索引存在arm_weight_file_seek_Vectors中。
        }
    }
    benchmark(model_name, in, opt); //to_arm_weight_file&arm_weight_file_seek_Vectors[weight_transform后的weight的索引]和DR_file_Vectors[表示file读取顺序(每个op在file中的起始位置)]
    if (file_bin)
    {
        WriteDataReaderFile(model_name); //将DR_file_Vectors[原始weight的file读取顺序(每个op在file中的起始位置)]，存储在"%NAME%.br.dat"
    }
    if(USE_PACK_ARM){
        if(ARM_W_TEST)
        {
            fclose(arm_weight_file);
            WritearmWeightDataReaderFile(model_name);//arm_weight_file_seek_Vectors[weight_transform后的weight的索引]写入“%NAME%.arm.br.dat"。
//            ReadarmWeightDataReaderFile(model_name);//读取文件“%NAME%.arm.br.dat"得到weight_transform后的weight的索引存在arm_weight_file_seek_Vectors中。
        }
        else{
            fclose(arm_weight_file_read);
            for(int i=0; i < 8; i++){
                fclose(arm_weight_file_reads[i]);
            }
        }
    }
#if NCNN_VULKAN
    if (use_vulkan_compute)
    {
        WriteSprivMapBinaryFile(model_name); //create_pipeline() spriv顺序
    }
#endif
    ARM_W_TEST = 0;
    TEST_ = 0;
    if(USE_PACK_ARM){
        arm_weight_file_read_init(model_name);
        ReadarmWeightDataReaderFile(model_name);
    }
//    benchmark(model_name, in, opt);
    std::cout<<"==========SAVE-TransWeight-TransWeightIdx-OriginWeightIdx[FINISH]==========\n"<<std::endl;
    std::cout<<"==========LOOP-DECIDE--cpuLittle_Vectors[4]&cpu_Vectors7--[START]=========="<<std::endl;

    for(int l=0; l<outerLoop; l++)
    {
        benchmark_biglittle(model_name, in, opt);
        loop_sum = sumRCtime(true);

//      loop_sum = sum_time;
        get_largecore_queue(l); //第一次不计算      //决定finish_f：
        if(finish_f){
            std::cout<<"==========[FINISH_F] r&p_cpu7=[] =========="<<std::endl;
            break;
        }


        get_queue_init(); //每个cpu队列的算子序号(根据get_time中测量的exec时间确定)

        for (int i = 0; i < interLoop; i++)
        {
            std::cout<<"================================outerLoop "<<l<<" interLoop "<<i<<"[START]============================"<<std::endl;

//            sum_time = 0;
//            if(USE_PACK_ARM){
//                arm_weight_file_read_init(model_name);
//                ReadarmWeightDataReaderFile(model_name);
//            }
            if(USE_PACK_ARM)
            {
                for(int it=0; it<8; it++){
                    fseek(arm_weight_file_reads[sched_getcpu()], 0, SEEK_SET);
                }
            }
            benchmark_new(model_name, in, opt);
//            if(USE_PACK_ARM){
//                fclose(arm_weight_file_read);
//                for(int i=0; i < 8; i++)
//                {
//                    fclose(arm_weight_file_reads[i]);
//                }
//            }

            loop_sum = sumRCtime(true);
            get_queue();
            std::cout<<"================================outerLoop "<<l<<" interLoop "<<i<<"[FINISH]============================"<<std::endl;
        }




        last_loop_sum = loop_sum;
        last_infer_time = infer_time;
        last_largecore_rc_time = largecore_rc_time;
        for (int i=0; i<CPU_NUMS; i++)
        {
            last_cpu_times[i] = cpu_times[i];
        }


    }
    WriteBinaryFile(model_name, use_vulkan_compute);
    if(USE_PACK_ARM){
        fclose(arm_weight_file_read);
        for(int i=0; i < 8; i++)
        {
            fclose(arm_weight_file_reads[i]);
        }
    }

//    double eee = ncnn::get_current_time();

#if NCNN_VULKAN
    delete g_blob_vkallocator;
    delete g_staging_vkallocator;
#endif // NCNN_VULKAN

    std::cout<<"==========LOOP-DECIDE--cpuLittle_Vectors[4]&cpu_Vectors7--[FINISH]=========="<<std::endl;
    printf("all latency: %fms\n", ncnn::get_current_time()-sss);
    return 0;
}
