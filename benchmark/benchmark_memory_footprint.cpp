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
#include <sys/resource.h>
#include <unistd.h>

#include <iostream>


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

#define __ARM_NEON 0
#define __CPU_MASK__ 1
#if __ARM_NEON
#include <arm_neon.h>
#endif

//
//export GOMP_CPU_AFFINITY="7 6 5 4";

char globalbinpath[256];
char gloableparampath[256];
#define THR 1
//#define ONE 1
//#define TWO 1
//#define FOR 1
size_t dr_cpu=0;//5;
size_t pipe_cpu=1;//6;
size_t infer_cpu=7;//0;
int warmUp = 0;

double start_t = 0;

//ncnn::Net net;
#ifdef __EMSCRIPTEN__
#define MODEL_DIR "/working/"
#else
#define MODEL_DIR ""
#endif
float GetWorkingSetSize() {
    long rss = 0L;
    FILE* fp = nullptr;
    if ((fp = fopen("/proc/self/statm", "r")) == nullptr) return 0;
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return ((size_t)rss * (size_t)sysconf(_SC_PAGESIZE)) / 1024.0f / 1024;
}

float GetPeakWorkingSetSize() {
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
    return static_cast<float>(rusage.ru_maxrss / 1024.0f);
}

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
        sleep(10);
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

    for (int i = 0; i < 0; i++)
    {
        double start = ncnn::get_current_time();

        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input(input_names[0], in);
            ex.extract(output_names[0], out);
            float* ptr = (float*)out.data;
            printf("out %f\n", *ptr);
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
//pthread_t t_infer;
//pthread_t t_pipe;
//pthread_t t_dr;
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

    //    float* ptr = (float*)out.data;
    //    printf("%f\n", *ptr);
    if(print)
    {
        infer_end = ncnn::get_current_time();
        infer_time = infer_end - infer_start;
        printf("for_skp_time=%f for_cal_time=%f\n", for_skp_time, for_cal_time);
        fprintf(stderr, "infer time =  %7.2f\n", infer_time);
    }

    //    pthread_exit(&t_infer);

}

void *infer_thread(void *args){
//   sleep(10);
//    fprintf(stderr, "start infer\n");
#ifdef __CPU_MASK__
    cpu_set_t mask;  //CPU核的集合
    cpu_set_t get;   //获取在集合中的CPU
    CPU_ZERO(&mask);    //置空
    CPU_SET(infer_cpu,&mask);   //设置亲和力值
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//设置线程CPU亲和力
    {
        printf("warning: could not set CPU affinity, continuing...\n");
    }
#endif


    auto *net=(ncnn::Net*)args;

    ncnn::Net net_in;
    net_in.opt = net->opt;
    char tparampath[256];
    sprintf(tparampath, MODEL_DIR "mobilenet_v3.param"); //GoogleNet  //AlexNet
    char tbinpath[256];
    sprintf(tbinpath, MODEL_DIR "mobilenet_v3.bin");
    net_in.load_param(tparampath);
    int open = net_in.load_model_dr(tbinpath);
    if (open<0)
    {
        printf("load file files\n");
        DataReaderFromEmpty dr;
        net_in.load_model_dr(dr);
    }
    net_in.load_model_pipe();
    //    printf("load p end %f\n", ncnn::get_current_time()-start_t);
    for (int i=0; i<10; i++)
    {
        ncnn::Mat top;
        //        conv7x7s2_pack1to4_neon(ncnn::Mat(227, 227, 3), top, ncnn::Mat(7,7, 3), ncnn::Mat(7,7, 3), net->opt);
        inference(net_in, ncnn::Mat(227, 227, 3), true);
        //        ncnn::do_forward_layer

        //        printf("infer p end %f\n", ncnn::get_current_time()-start_t);
    }
    ncnn::current_layer_idx_f2p = 0;
    ncnn::current_layer_idx_p2i = 0;
    infer_start = 0;
    pipe_start = 0;
    for_cal_time = 0;
    for_skp_time = 0;
    read_syn = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};
    create_syn = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};
    infer_syn = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};


    //        int open = net->load_model_dr(globalbinpath);
    //        if (open<0)
    //        {
    //            printf("load file files\n");
    //            DataReaderFromEmpty dr;
    //            net->load_model_dr(dr);
    //        }
    //        net->load_model_pipe();
    //        for (int i=0; i<2; i++)
    //        {
    //            inference(*net, ncnn::Mat(227, 227, 3));
    //        }
    //    ncnn::current_layer_idx_f2p = 0;
    //    ncnn::current_layer_idx_p2i = 0;
    //    infer_start = 0;
    //    pipe_start = 0;
    //    for_cal_time = 0;
    //    for_skp_time = 0;

    //    net_in.load_model(globalbinpath);
    //    net_in.load_model_pipe();
    //    inference(net_in, ncnn::Mat(227, 227, 3));

    //    net.load_param(gloableparampath);
    //


    pthread_mutex_lock(&param_lock);
    param_finish = 100;
    param_finish_1 = 100;
    pthread_mutex_unlock(&param_lock);
    pthread_cond_signal(&param_cond);
    pthread_cond_signal(&param_cond_1);


    pthread_cond_signal(&param_cond_cpu0);
    pthread_cond_signal(&param_cond_cpu1);
    pthread_cond_signal(&param_cond_cpu2);
    pthread_cond_signal(&param_cond_cpu3);

    printf("param finish_____________\n");

    //    pthread_mutex_lock(&param_lock);
    //    while (param_finish_1 == 0 ){
    //        pthread_cond_wait(&param_cond_1, &param_lock);
    //    }
    //    pthread_mutex_unlock(&param_lock);

    //    int open = net->load_model_dr(globalbinpath);
    //    if (open<0)
    //    {
    //        printf("load file files\n");
    //        DataReaderFromEmpty dr;
    //        net->load_model_dr(dr);
    //    }
    //    net->load_model_pipe();
    //    net->load_model_pipe();
    //    net->load_model_pipe();
    //    net->load_model_pipe();
    //    net->load_model_pipe();
    //    inference(*net, ncnn::Mat(227, 227, 3));
    //    inference(*net, ncnn::Mat(227, 227, 3));
    //    inference(*net, ncnn::Mat(227, 227, 3));

    //    net->load_param(gloableparampath);
    printf("__________inf tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    inference(*net, ncnn::Mat(227, 227, 3), true);
    //    inference(net, ncnn::Mat(227, 227, 3));
    //    inference(*stu1, ncnn::Mat(227, 227, 3));
    //    inference(*stu1, ncnn::Mat(227, 227, 3));
    //    inference(*stu1, ncnn::Mat(227, 227, 3));
    //    inference(*stu1, ncnn::Mat(227, 227, 3));
    //    inference(*stu1, ncnn::Mat(227, 227, 3));
    //    inference(*stu1, ncnn::Mat(227, 227, 3));
    //    inference(*stu1, ncnn::Mat(227, 227, 3));
    //    inference(*stu1, ncnn::Mat(227, 227, 3));


    //    for(int ii=0; ii<0;ii++)
    //    {
    //        const std::vector<const char*>& input_names__ = net->input_names();
    //        const std::vector<const char*>& output_names__ = net->output_names();
    //
    //        ncnn::Mat out__;
    //
    //        infer_start = ncnn::get_current_time();
    //
    //        ncnn::Extractor ex__ = net->create_extractor();
    //        ex__.input(input_names__[0], ncnn::Mat(227, 227, 3));
    //        ex__.extract(output_names__[0], out__);
    //
    //        //        float* ptr = (float*)out__.data;
    //        //        printf("%f\n", *ptr);
    //
    //        infer_end = ncnn::get_current_time();
    //        infer_time = infer_end - infer_start;
    //        fprintf(stderr, "2d infer time =  %7.2f\n", infer_time);
    //    }

    return 0;
    //    printf("infer fork %d", fork());
    //    while(1)
    //        {
    //            sleep(1);
    //            printf("inf tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    //        }
}

void *dr_thread(void *args){
#ifdef __CPU_MASK__
    cpu_set_t mask;  //CPU核的集合
    cpu_set_t get;   //获取在集合中的CPU
    CPU_ZERO(&mask);    //置空
    CPU_SET(dr_cpu,&mask);   //设置亲和力值
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//设置线程CPU亲和力
    {
        printf("warning: could not set CPU affinity, continuing...\n");
    }
#endif

    //    fprintf(stderr, "start dr\n");
    auto *net=(ncnn::Net*)args;
    //    printf("drr tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());

    //    net.load_param(gloableparampath);

    //    pthread_mutex_lock(&param_lock);
    //    param_finish = 100;
    //    param_finish_1 = 100;
    //    pthread_mutex_unlock(&param_lock);
    //    pthread_cond_signal(&param_cond);
    //    pthread_cond_signal(&param_cond_1);
    //    printf("param finish_____________\n");

    pthread_mutex_lock(&param_lock);
    while (param_finish_1 == 0 ){
        pthread_cond_wait(&param_cond_1, &param_lock);
    }
    pthread_mutex_unlock(&param_lock);
    //        net.load_param(gloableparampath);

    dr_start = ncnn::get_current_time();

    //    printf("%d\n", fork());

    //    ncnn::Layer* layer = net->layers()[0];
    //    net->load_model_dr_layer(gloableparampath, 0);

    int open = net->load_model_dr(globalbinpath);
    if (open<0)
    {
        printf("load file files\n");
        DataReaderFromEmpty dr;
        net->load_model_dr(dr);
    }
    //
    //
    //    //    DataReaderFromEmpty dr;
    //    //    net->load_model_dr(dr);
    //
    dr_end = ncnn::get_current_time();
    dr_time = dr_end - dr_start;

    fprintf(stderr, "load model = %7.2f\n", dr_time);
    //    printf("dr fork %d", fork());
    //    while(1)
    //    {
    //        sleep(1);
    //        printf("drr tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    //    }

    //    sleep(200000);
    //    while(1);
    return 0;
    //    pthread_exit(&t_dr);


    //    inference(*net, ncnn::Mat(227, 227, 3));
    //    fprintf(stderr, "inference\n");
}

void *pipe_thread(void *args){

    //    pthread_t t_pipe;
    //pthread_t t_dr;

#ifdef __CPU_MASK__
    cpu_set_t mask;  //CPU核的集合
    cpu_set_t get;   //获取在集合中的CPU
    CPU_ZERO(&mask);    //置空
    CPU_SET(pipe_cpu,&mask);   //设置亲和力值
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//设置线程CPU亲和力
    {
        printf("warning: could not set CPU affinity, continuing...\n");
    }
#endif
    //    fprintf(stderr, "start pipe\n");
    //    net->load_param(gloableparampath);

    //    printf("pip tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    //    pipe_start = ncnn::get_current_time();
    //    sleep(1);

    pthread_mutex_lock(&param_lock);
    while (param_finish == 0 ){
        pthread_cond_wait(&param_cond, &param_lock);
    }
    pthread_mutex_unlock(&param_lock);
    //    net.load_param(gloableparampath);

    auto *net=(ncnn::Net*)args;

    //    net->load_model_dr_layer(gloableparampath, 1);

    net->load_model_pipe();

    pipe_end = ncnn::get_current_time();
    pipe_time = pipe_end - pipe_start;

    //    fprintf(stderr, "load pipeline = %7.2f\n", pipe_time);
    //    printf("pp fork %d", fork());
    //    while(1)
    //    {
    //        sleep(1);
    //        printf("pip tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    //    }

    //    sleep(200000);
    //    while(1);
    return 0;
    //    pthread_exit(&t_pipe);

    //    inference(*net, ncnn::Mat(227, 227, 3));
    //    fprintf(stderr, "inference\n");
}

/////////////////////////////////////////////////////////////////////////////
DataReaderFromEmpty drp;
ncnn::ModelBinFromDataReader mb(drp);

#define CPU_NUMS 4
std::vector<int> cpu_Vectors[CPU_NUMS];

std::vector<int> cpu0_Vector, cpu1_Vector, cpu2_Vector, cpu3_Vector, cpu7_Vector;

void *thread_cpu4567(void *args)
{
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

    auto* net = (ncnn::Net*)args;
    if(warmUp)
    {
        ncnn::Net net_in;
        net_in.opt = net->opt;
        char tparampath[256];
        sprintf(tparampath, MODEL_DIR "mobilenet_v3.param"); //GoogleNet  //AlexNet
        char tbinpath[256];
        sprintf(tbinpath, MODEL_DIR "mobilenet_v3.bin");
        net_in.load_param(tparampath);
        int open = net_in.load_model_dr(tbinpath);
        if (open < 0)
        {
            printf("load file files\n");
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

    pthread_mutex_lock(&param_lock);
    param_finish_1 = 100;
    pthread_mutex_unlock(&param_lock);
    pthread_cond_signal(&param_cond_cpu3);
    pthread_cond_signal(&param_cond_cpu2);
    pthread_cond_signal(&param_cond_cpu1);
    pthread_cond_signal(&param_cond_cpu0);
    pthread_cond_signal(&param_cond_cpu7);
    pthread_cond_signal(&param_cond_cpu6);
    pthread_cond_signal(&param_cond_cpu5);
    pthread_cond_signal(&param_cond_cpu4);


    printf("param finish_____________\n");
    printf("__________t_cpu4567 tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    double start_time = ncnn::get_current_time();
    /*read*/
    printf("load file %s\n", globalbinpath);
    FILE* fp = fopen(globalbinpath, "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen %s failed", globalbinpath);
        //        return 0;
    }
    else
    {
        ncnn::DataReaderFromStdio dr(fp);
        if (net->layers().empty())
        {
            NCNN_LOGE("network graph not ready");
            //        return 0;
        }
        ncnn::ModelBinFromDataReader mb1(dr);

        mb = mb1;
    }
    /*create*/
    double s = ncnn::get_current_time();
    net->load_model_layer(mb, 0);
    net->load_pipe_layer(0);
    net->load_model_layer(mb, 1);
    syn_act(read_syn, 2);
    net->load_pipe_layer(1);
    syn_act(infer_syn, 1);
    int list_len = cpu7_Vector.size();
    double cal_time= 0;
    for(int ii=0; ii<list_len; ii++){
        int i = cpu7_Vector[ii];
        net->load_model_layer(mb, i);
        net->load_pipe_layer(i);
        syn_act(infer_syn, i);
    }

    double start_time_inf = ncnn::get_current_time();
    inference(*net, ncnn::Mat(227, 227, 3), true);
    double end_time = ncnn::get_current_time();
    printf("\n===================================================================================================rc=%f infer=%f =tttime = %f\n", start_time_inf-s, end_time - start_time_inf, end_time - start_time);
    if (fp)
    {
        fclose(fp);
    }

    param_finish_1 = 0;
    return 0;
}
void *thread_cpu3(void *args){

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
    printf("__________t_cpu3    tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    double start_time = ncnn::get_current_time();

    auto *net=(ncnn::Net*)args;
    int layer_count = (int)net->layers().size();
    int start_i=2;
    for(int i=start_i; i<layer_count;){//} i=i+4){
        //        syn_wait(read_syn, i);
        double s = ncnn::get_current_time();
        net->load_model_layer(mb, i);
        //        if(i+1<layer_count)
        //            syn_act(read_syn, i+1);
        net->load_pipe_layer(i);

        if(timeshow)
        {
            double e = ncnn::get_current_time();
            printf("%d,%f\n", i, e - s);
        }

        syn_act(infer_syn, i);

        pthread_mutex_lock(&next_layer_lock);
        i = select_next_layer();
        pthread_mutex_unlock(&next_layer_lock);
    }
    double end_time = ncnn::get_current_time();
    printf("\n==============================================================================================cpu3_time = %ff\n", end_time - start_time);
    return 0;
}

typedef struct net_cid{
    int cpu_id;
    ncnn::Net* net;
}net_cid;
void *thread_list(void *args){
    pthread_mutex_lock(&param_lock);
    while (param_finish_1 == 0 ){
        //        printf("=================\n");
        pthread_cond_wait(&param_cond_cpu3, &param_lock);
    }
    pthread_mutex_unlock(&param_lock);

    auto *net_cid_ = (net_cid*)args;

#ifdef __CPU_MASK__
    cpu_set_t mask;  //CPU核的集合
    CPU_ZERO(&mask);    //置空
    CPU_SET(net_cid_->cpu_id,&mask);   //设置亲和力值
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//设置线程CPU亲和力
    {
        printf("warning: could not set CPU affinity, continuing...\n");
    }
#endif
    printf("__________t_cpu%d    tid=%ld,cpu=%d\n",net_cid_->cpu_id, pthread_self(), sched_getcpu());
    double start_time = ncnn::get_current_time();


    //    auto *net=(ncnn::Net*)args;
    int list_len = cpu_Vectors[net_cid_->cpu_id].size();
    double cal_time= 0;
    for(int ii=0; ii<list_len; ii++){
        int i = cpu_Vectors[net_cid_->cpu_id][ii];

        double s = ncnn::get_current_time();
        net_cid_->net->load_model_layer(mb, i);
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
    printf("\n==============================================================================================list_cpu%d_time = %f %f\n", sched_getcpu(), end_time - start_time, cal_time);
    return 0;
}

void *thread_list_cpu3(void *args){
    pthread_mutex_lock(&param_lock);
    while (param_finish_1 == 0 ){
        //        printf("=================\n");
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
    printf("__________t_cpu3    tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    double start_time = ncnn::get_current_time();

    auto *net=(ncnn::Net*)args;
    int list_len = cpu3_Vector.size();
    double cal_time= 0;
    for(int ii=0; ii<list_len; ii++){
        int i = cpu3_Vector[ii];

        double s = ncnn::get_current_time();
        net->load_model_layer(mb, i);
        net->load_pipe_layer(i);
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
    printf("\n==============================================================================================list_cpu3_time = %f %f\n", end_time - start_time, cal_time);
    return 0;
}

void *thread_list_cpu2(void *args){

    pthread_mutex_lock(&param_lock);
    while (param_finish_1 == 0 ){
        pthread_cond_wait(&param_cond_cpu2, &param_lock);
    }
    pthread_mutex_unlock(&param_lock);

#ifdef __CPU_MASK__
    cpu_set_t mask;  //CPU核的集合
    CPU_ZERO(&mask);    //置空
    CPU_SET(2,&mask);   //设置亲和力值
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//设置线程CPU亲和力
    {
        printf("warning: could not set CPU affinity, continuing...\n");
    }
#endif
    printf("__________t_cpu2    tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    double start_time = ncnn::get_current_time();

    auto *net=(ncnn::Net*)args;
    int list_len = cpu2_Vector.size();
    double cal_time= 0;
    for(int ii=0; ii<list_len; ii++){
        int i = cpu2_Vector[ii];
        double s = ncnn::get_current_time();
        net->load_model_layer(mb, i);
        net->load_pipe_layer(i);
        double e = ncnn::get_current_time();
        cal_time+= e-s;
        if(timeshow)
        {
            printf("%d,%f\n", i, e - s);
        }
        syn_act(infer_syn, i);
    }
    double end_time = ncnn::get_current_time();
    printf("\n==============================================================================================list_cpu2_time = %f %f\n", end_time - start_time, cal_time);
    return 0;
}

void *thread_list_cpu1(void *args){

    pthread_mutex_lock(&param_lock);
    while (param_finish_1 == 0 ){
        pthread_cond_wait(&param_cond_cpu1, &param_lock);
    }
    pthread_mutex_unlock(&param_lock);

#ifdef __CPU_MASK__
    cpu_set_t mask;  //CPU核的集合
    CPU_ZERO(&mask);    //置空
    CPU_SET(1,&mask);   //设置亲和力值
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//设置线程CPU亲和力
    {
        printf("warning: could not set CPU affinity, continuing...\n");
    }
#endif
    printf("__________t_cpu1    tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    double start_time = ncnn::get_current_time();

    auto *net=(ncnn::Net*)args;
    int list_len = cpu1_Vector.size();
    double cal_time= 0;
    for(int ii=0; ii<list_len; ii++){
        int i = cpu1_Vector[ii];
        double s = ncnn::get_current_time();
        net->load_model_layer(mb, i);
        net->load_pipe_layer(i);
        double e = ncnn::get_current_time();
        cal_time+= e-s;
        if(timeshow)
        {
            printf("%d,%f\n", i, e - s);
        }
        syn_act(infer_syn, i);
    }
    double end_time = ncnn::get_current_time();
    printf("\n==============================================================================================list_cpu1_time = %f %f\n", end_time - start_time, cal_time);
    return 0;
}

void *thread_list_cpu0(void *args){

    pthread_mutex_lock(&param_lock);
    while (param_finish_1 == 0 ){
        pthread_cond_wait(&param_cond_cpu0, &param_lock);
    }
    pthread_mutex_unlock(&param_lock);

#ifdef __CPU_MASK__
    cpu_set_t mask;  //CPU核的集合
    CPU_ZERO(&mask);    //置空
    CPU_SET(0,&mask);   //设置亲和力值
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//设置线程CPU亲和力
    {
        printf("warning: could not set CPU affinity, continuing...\n");
    }
#endif
    printf("__________t_cpu0    tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    double start_time = ncnn::get_current_time();

    auto *net=(ncnn::Net*)args;
    int list_len = cpu0_Vector.size();
    double cal_time= 0;
    for(int ii=0; ii<list_len; ii++){
        int i = cpu0_Vector[ii];
        double s = ncnn::get_current_time();
        net->load_model_layer(mb, i);
        net->load_pipe_layer(i);
        double e = ncnn::get_current_time();
        cal_time+= e-s;
        if(timeshow)
        {
            printf("%d,%f\n", i, e - s);
        }
        syn_act(infer_syn, i);
    }
    double end_time = ncnn::get_current_time();
    printf("\n==============================================================================================list_cpu0_time = %f %f\n", end_time - start_time, cal_time);
    return 0;
}


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



    printf("start %f\n", ncnn::get_current_time()-start_t);
    printf("\n==s====================\n");
    printf("\n=========s=============\n");
    //        std::thread t2(pipe_thread, (void*)&net);
    //        std::thread t1(dr_thread, (void*)&net);
    //        std::thread t3(infer_thread, (void*)&net);
    //
    //        t2.join();
    //        t1.join();
    //        t3.join();



    std::thread t_4567(thread_cpu4567, (void*)&net);

    net_cid net_cid3 =  {3, &net};
    std::thread t_3(thread_list, (void*)&net_cid3);
    net_cid net_cid2 =  {2, &net};
    std::thread t_2(thread_list, (void*)&net_cid2);
    net_cid net_cid1 =  {1, &net};
    std::thread t_1(thread_list, (void*)&net_cid1);
    net_cid net_cid0 =  {0, &net};
    std::thread t_0(thread_list, (void*)&net_cid0);

//    std::thread t_3(thread_list_cpu3, (void*)&net);
//    std::thread t_2(thread_list_cpu2, (void*)&net);
//    std::thread t_1(thread_list_cpu1, (void*)&net);
//    std::thread t_0(thread_list_cpu0, (void*)&net);

//    thread_cpu4567((void*)&net);
    t_4567.join();
    t_3.join();
    t_2.join();
    t_1.join();
    t_0.join();





        printf("inference: %.3fMB, peak: %.3fMB\n", GetWorkingSetSize(),
               GetPeakWorkingSetSize());
    printf("_____________________________________________________________________________end %f\n", ncnn::get_current_time()-start_t);





    //    double a_[3] = {dr_end, pipe_end, infer_end};
    //    double *b_;
    //    b_ = std::max_element(a_, a_+3);
    //    double max_ = *b_;
    //
    //    double a1_[3] = {dr_start, pipe_start, infer_start};
    //    double *b1_;
    //    b1_ = std::min_element(a1_, a1_+3);
    //    double min_ = *b1_;
    //
    //    double tt_time_ = max_ - min_;
    //
    //    printf("total time = %7.2f\n", tt_time_);
    //    printf("load start=%7.2f, end=%7.2f, tt=%7.2f\n", dr_start-min_, dr_end-min_, dr_end-dr_start);
    //    printf("pipe start=%7.2f, end=%7.2f, tt=%7.2f\n", pipe_start-min_, pipe_end-min_, pipe_end-pipe_start);
    //    printf("infe start=%7.2f, end=%7.2f, tt=%7.2f\n", infer_start-min_, infer_end-min_, infer_end-infer_start);

    //    }



//    printf("==============================================\n");
//
//    // inference
//    //    sleep(0);
//
//
//    //    net.load_param(parampath);
//    for(int ii=0; ii<0;ii++)
//    {
//        const std::vector<const char*>& input_names__ = net.input_names();
//        const std::vector<const char*>& output_names__ = net.output_names();
//
//        ncnn::Mat out__;
//
//        infer_start = ncnn::get_current_time();
//
//        ncnn::Extractor ex__ = net.create_extractor();
//        ex__.input(input_names__[0], in);
//        ex__.extract(output_names__[0], out__);
//
//        //        float* ptr = (float*)out__.data;
//        //        printf("%f\n", *ptr);
//
//        infer_end = ncnn::get_current_time();
//        infer_time = infer_end - infer_start;
//        fprintf(stderr, "2d infer time =  %7.2f\n", infer_time);
//    }

    //        pthread_exit(NULL);

}

void ReadBinaryFile( const char* comment)
{
    char path[256];
    sprintf(path, MODEL_DIR "%s.dat", comment);
    std::ifstream isData(path, std::ios_base::in | std::ios_base::binary);
    if (isData)
    {
        for (int i=0; i<CPU_NUMS; i++){
            int n = CPU_NUMS - i-1;
            int cpu_Vsize;
            isData.read(reinterpret_cast<char *>(&cpu_Vsize),sizeof(cpu_Vsize));
            cpu_Vectors[n].resize(cpu_Vsize);
            isData.read(reinterpret_cast<char *>(cpu_Vectors[n].data()), cpu_Vsize *sizeof(int) );

            printf("%d:{", n);
            for(int i : cpu_Vectors[n]){
                printf("%d,", i);
            }
            printf("}\n");
        }
        int cpu7Vsize;
        isData.read(reinterpret_cast<char *>(&cpu7Vsize),sizeof(cpu7Vsize));
        cpu7_Vector.resize(cpu7Vsize);
        isData.read(reinterpret_cast<char *>(cpu7_Vector.data()), cpu7Vsize *sizeof(int) );

        printf("7:{");
        for(int i : cpu7_Vector){
            printf("%d,", i);
        }
        printf("}\n");
    }
    else
    {
        printf("ERROR: Cannot open file 数据.dat");
    }
}

void ReadBinaryFile_( const char* comment)
{
    double start = ncnn::get_current_time();

    char path[256];
    sprintf(path, MODEL_DIR "%s.dat", comment);
    std::ifstream isData(path, std::ios_base::in | std::ios_base::binary);
    if (isData)
    {
        int cpu3Vsize;
        isData.read(reinterpret_cast<char *>(&cpu3Vsize),sizeof(cpu3Vsize));
        cpu3_Vector.resize(cpu3Vsize);
        isData.read(reinterpret_cast<char *>(cpu3_Vector.data()), cpu3Vsize *sizeof(int) );

        int cpu2Vsize;
        isData.read(reinterpret_cast<char *>(&cpu2Vsize),sizeof(cpu2Vsize));
        cpu2_Vector.resize(cpu2Vsize);
        isData.read(reinterpret_cast<char *>(cpu2_Vector.data()), cpu2Vsize *sizeof(int) );

        int cpu1Vsize;
        isData.read(reinterpret_cast<char *>(&cpu1Vsize),sizeof(cpu1Vsize));
        cpu1_Vector.resize(cpu1Vsize);
        isData.read(reinterpret_cast<char *>(cpu1_Vector.data()), cpu1Vsize *sizeof(int) );

        int cpu0Vsize;
        isData.read(reinterpret_cast<char *>(&cpu0Vsize),sizeof(cpu0Vsize));
        cpu0_Vector.resize(cpu0Vsize);
        isData.read(reinterpret_cast<char *>(cpu0_Vector.data()), cpu0Vsize *sizeof(int) );

        int cpu7Vsize;
        isData.read(reinterpret_cast<char *>(&cpu7Vsize),sizeof(cpu7Vsize));
        cpu7_Vector.resize(cpu7Vsize);
        isData.read(reinterpret_cast<char *>(cpu7_Vector.data()), cpu7Vsize *sizeof(int) );

        double end = ncnn::get_current_time();
        printf("load time:%f\n", end- start);
        printf("0:{");
        for(int i : cpu0_Vector){
            printf("%d,", i);
        }
        printf("}\n");
        printf("1:{");
        for(int i : cpu1_Vector){
            printf("%d,", i);
        }
        printf("}\n");
        printf("2:{");
        for(int i : cpu2_Vector){
            printf("%d,", i);
        }
        printf("}\n");
        printf("3:{");
        for(int i : cpu3_Vector){
            printf("%d,", i);
        }
        printf("}\n");
        printf("7:{");
        for(int i : cpu7_Vector){
            printf("%d,", i);
        }
        printf("}\n");
    }
    else
    {
        printf("ERROR: Cannot open file 数据.dat");
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

    if (argc >= 2)
    {
        loop_count = atoi(argv[1]);
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


    char model_name[] = "GoogleNet";
    ncnn::Mat in = ncnn::Mat(224, 224, 3);
    ReadBinaryFile(model_name);

    printf("init: %.3fMB, peak: %.3fMB\n", GetWorkingSetSize(),
               GetPeakWorkingSetSize());
    benchmark_new(model_name, in, opt);


//    benchmark(model_name, in, opt);

    printf("inference: %.3fMB, peak: %.3fMB\n", GetWorkingSetSize(),
                    GetPeakWorkingSetSize());
    //        benchmark("GoogleNet", ncnn::Mat(224, 224, 3), opt);
    // run
    //        benchmark_new("AlexNet", ncnn::Mat(227, 227, 3), opt);
    //        benchmark_new("GoogleNet", ncnn::Mat(227, 227, 3), opt);
    //        benchmark_new("MobileNet", ncnn::Mat(224, 224, 3), opt);
    //        benchmark_new("MobileNet", ncnn::Mat(224, 224, 3), opt);
    //        benchmark_new("MobileNetV2", ncnn::Mat(224, 224, 3), opt);
    //        benchmark_new("resnet18", ncnn::Mat(224, 224, 3), opt);

    //    benchmark("mobilenet_int8", ncnn::Mat(224, 224, 3), opt);
    //    benchmark("googlenet_int8", ncnn::Mat(224, 224, 3), opt);
    //    benchmark_new("resnet18_int8", ncnn::Mat(224, 224, 3), opt);
    //    sleep(50);
    //    sleep(100);

    //    benchmark("squeezenet", ncnn::Mat(227, 227, 3), opt);
    //
    //    benchmark("squeezenet_int8", ncnn::Mat(227, 227, 3), opt);
    //
    //    benchmark("mobilenet", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("mobilenet_int8", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("mobilenet_v2", ncnn::Mat(224, 224, 3), opt);
    //
    //    // benchmark("mobilenet_v2_int8", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("mobilenet_v3", ncnn::Mat(224, 224, 3), opt);
    //
    //        benchmark("shufflenet", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("shufflenet_v2", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("mnasnet", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("proxylessnasnet", ncnn::Mat(224, 224, 3), opt);
    //
    //        benchmark("efficientnet_b0", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("efficientnetv2_b0", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("regnety_400m", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("blazeface", ncnn::Mat(128, 128, 3), opt);
    //
    //    benchmark("googlenet", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("googlenet_int8", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("resnet18", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("resnet18_int8", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("alexnet", ncnn::Mat(227, 227, 3), opt);
    //
    //    benchmark("vgg16", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("vgg16_int8", ncnn::Mat(224, 224, 3), opt);
    //
    //        benchmark("resnet50", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("resnet50_int8", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("squeezenet_ssd", ncnn::Mat(300, 300, 3), opt);
    //
    //    benchmark("squeezenet_ssd_int8", ncnn::Mat(300, 300, 3), opt);
    //
    //    benchmark("mobilenet_ssd", ncnn::Mat(300, 300, 3), opt);
    //
    //    benchmark("mobilenet_ssd_int8", ncnn::Mat(300, 300, 3), opt);
    //
    //    benchmark("mobilenet_yolo", ncnn::Mat(416, 416, 3), opt);
    //
    //    benchmark("mobilenetv2_yolov3", ncnn::Mat(352, 352, 3), opt);
    //
    //    benchmark("yolov4-tiny", ncnn::Mat(416, 416, 3), opt);

#if NCNN_VULKAN
    delete g_blob_vkallocator;
    delete g_staging_vkallocator;
#endif // NCNN_VULKAN

    //    printf("end %f\n", ncnn::get_current_time()-start_t);
    return 0;
}
