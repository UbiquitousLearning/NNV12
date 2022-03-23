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
size_t dr_cpu=0;//5;
size_t pipe_cpu=1;//6;
size_t infer_cpu=7;//0;

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

double largecore_rc_time=0;
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
        printf("for_skp_time=%f for_cal_time=%f\t", for_skp_time, for_cal_time);
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


    printf("__________inf tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    inference(*net, ncnn::Mat(227, 227, 3), true);

    return 0;
}

void *pipe_infer_thread(void *args){
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
    printf("__________pipe tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    net->load_model_pipe();
    pipe_end = ncnn::get_current_time();
    pipe_time = pipe_end - pipe_start;
    printf("__________inf tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    inference(*net, ncnn::Mat(227, 227, 3));
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
std::vector<double> rc_times(1000, -1);
DataReaderFromEmpty drp;
ncnn::ModelBinFromDataReader mb(drp);


#define CPU_NUMS 4
std::vector<int> cpu_Vectors[CPU_NUMS];
double cpu_times[CPU_NUMS];
pthread_cond_t param_cond_cpus[CPU_NUMS];

void *thread_cpu4567(void *args){
#ifdef __EMSCRIPTEN__
#define MODEL_DIR "/working/"
#else
#define MODEL_DIR ""
#endif
    param_finish_1 = 0;
#ifdef __CPU_MASK__
    cpu_set_t mask;  //CPU核的集合
    CPU_ZERO(&mask);    //置空
    CPU_SET(7,&mask);   //设置亲和力值
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//设置线程CPU亲和力
    {
        printf("warning: could not set CPU affinity, continuing...\n");
    }
#endif
    finish_set_init();

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
        DataReaderFromEmpty dr;
        net_in.load_model_dr(dr);
    }
    net_in.load_model_pipe();
    for (int i=0; i<10; i++)
    {
        ncnn::Mat top;
        inference(net_in, ncnn::Mat(227, 227, 3), false);

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
    double s0 = ncnn::get_current_time();
    net->load_model_layer(mb, 0);
    net->load_pipe_layer(0);
    double e0 = ncnn::get_current_time();
    rc_times[0] = e0-s0;

    double s1 = ncnn::get_current_time();
    net->load_model_layer(mb, 1);
    syn_act(read_syn, 2);
    net->load_pipe_layer(1);
    double e1 = ncnn::get_current_time();
    rc_times[1] = e1-s1;
    syn_act(infer_syn, 1);

    //    int list_len = sizeof(cpu7_list)/ sizeof(cpu7_list[0]);
    //    double cal_time= 0;
    //    for(int ii=0; ii<list_len; ii++){
    //        int i = cpu7_list[ii];
    for(int i: cpu7_vector){
        //        double s = ncnn::get_current_time();
        net->load_model_layer(mb, i);
        net->load_pipe_layer(i);
        //        double e = ncnn::get_current_time();
        //        rc_times[i] = e-s;
        syn_act(infer_syn, i);
    }

    double start_time_inf = ncnn::get_current_time();
    inference(*net, ncnn::Mat(227, 227, 3), true);
    double end_time = ncnn::get_current_time();
    largecore_rc_time = start_time_inf-s0;
    printf("\n===================================================================================================rc=%f infer=%f time = %f\n", start_time_inf-s0, end_time - start_time_inf, end_time - start_time);
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
    for(int i=start_i; i<layer_count;){
        double s = ncnn::get_current_time();
        net->load_model_layer(mb, i);
        net->load_pipe_layer(i);
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
    printf("\n==============================================================================================cpu3_time = %ff\n", end_time - start_time);
    return 0;
}


typedef struct net_cid{
    int cpu_id;
    int cpu_set;
    ncnn::Net* net;
}net_cid;
void *thread_list(void *args){

    auto *net_cid_ = (net_cid*)args;

    pthread_mutex_lock(&param_lock);
    while (param_finish_1 == 0 ){
        //        printf("=================\n");
        pthread_cond_wait(&param_cond_cpus[net_cid_->cpu_id], &param_lock);
        //        sleep(0);
    }
    pthread_mutex_unlock(&param_lock);

#ifdef __CPU_MASK__
    cpu_set_t mask;  //CPU核的集合
    CPU_ZERO(&mask);    //置空
    CPU_SET(net_cid_->cpu_set,&mask);   //设置亲和力值
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//设置线程CPU亲和力
    {
        printf("warning: could not set CPU affinity, continuing...\n");
    }
#endif
    printf("__________t%d_cpu%d    tid=%ld,cpu=%d\n",net_cid_->cpu_id, net_cid_->cpu_set, pthread_self(), sched_getcpu());
    double start_time = ncnn::get_current_time();


    //        auto net=net_cid_->net;
    int list_len = cpu_Vectors[net_cid_->cpu_id].size();
    double cal_time= 0;
    for(int ii=0; ii<list_len; ii++){
        int i = cpu_Vectors[net_cid_->cpu_id][ii];
        double s = ncnn::get_current_time();
        net_cid_->net->load_model_layer(mb, i);
        net_cid_->net->load_pipe_layer(i);
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
    printf("\n==============================================================================================list_cpu%d_time = %f %f\n", sched_getcpu(), end_time - start_time, cal_time);
    cpu_times[net_cid_->cpu_id] = end_time - start_time;
    return 0;
}


void benchmark_new(const char* comment, const ncnn::Mat& _in, const ncnn::Option& opt)
{
    //    rc_times.clear();

    layer_next=5;
    timeshow=0;

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
    //    printf("rc_times_len=%zu\n", rc_times.size());

    std::thread t_4567(thread_cpu4567, (void*)&net);
    net_cid net_cid3 =  {3, 3, &net};
    std::thread t_3(thread_list, (void*)&net_cid3);
    net_cid net_cid2 =  {2, 2, &net};
    std::thread t_2(thread_list, (void*)&net_cid2);
    net_cid net_cid1 =  {1, 1, &net};
    std::thread t_1(thread_list, (void*)&net_cid1);
    net_cid net_cid0 =  {0, 0, &net};
    std::thread t_0(thread_list, (void*)&net_cid0);
    //    std::thread t_3(thread_list_cpu3, (void*)&net);
    //    std::thread t_2(thread_list_cpu2, (void*)&net);
    //    std::thread t_1(thread_list_cpu1, (void*)&net);
    //    std::thread t_0(thread_list_cpu0, (void*)&net);

    t_4567.join();
    t_3.join();
    t_2.join();
    t_1.join();
    t_0.join();

}


void get_time(const char* comment, const ncnn::Mat& _in, const ncnn::Option& opt)
{
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
    //    printf("rc_times_len=%zu\n", rc_times.size());

    std::thread t_4567(thread_cpu4567, (void*)&net);
    std::thread t_3(thread_cpu3, (void*)&net);
    t_4567.join();
    t_3.join();
}
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

double loop_sum=0, last_loop_sum=0, last_skp_time=0, last_cal_time=0, last_largecore_rc_time=0;
double last_cpu_times[CPU_NUMS];
double large_rc_need=0;
double eta=0.125;
int finish_f=0;


void get_largecore_queue(int cnt=0){

    if(cnt==0){
        return;
    }


    printf("=====================sum=%f rc_time=%f for_skp_time=%f for_cal_time=%f,  \n", last_loop_sum+4, last_largecore_rc_time, last_skp_time, last_cal_time);
    for (double last_cpu_time:last_cpu_times)
    {
        printf("%f\n", last_cpu_time);
    }
    //    printf("%f %f %f %F\n", last_cpu3_time, last_cpu2_time, last_cpu1_time, last_cpu0_time);

    //    double cpu_times[]={last_cpu0_time, last_cpu1_time, last_cpu2_time, last_cpu3_time};
    double max_cpu_time = *std::max_element(last_cpu_times,last_cpu_times+4);
    double diff_time = max_cpu_time+1 -(last_largecore_rc_time + last_cal_time);


    printf("diff %f\n", diff_time);
    if(diff_time<0&& cpu7_vector.size()==0){
        finish_f=1;
        return;
    }
    large_rc_need +=diff_time*eta;
    printf("%f diff %f\n", large_rc_need, diff_time);

    std::vector<int> idx_vector;
    for(int i=2; i<rc_times.size(); i++) {
        idx_vector.push_back(i);
    }

    int len = idx_vector.size();
    int i, j; double temp;
    for (i = 0; i < len - 1; i++)
        for (j = 0; j < len - 1 - i; j++)
            if (rc_times[idx_vector[j]] < rc_times[idx_vector[j + 1]]) {
                temp = idx_vector[j];
                idx_vector[j] = idx_vector[j + 1];
                idx_vector[j + 1] = temp;
            }
    //    for(int i: idx_vector){
    //        printf("%d,", i);
    //    }
    //    printf("\n");

    std::vector<int> cpu7_vector_;
    double ssum=0;
    for(int idx :idx_vector){
        if(ssum + rc_times[idx]<=large_rc_need)
        {
            ssum+=rc_times[idx];
            cpu7_vector_.push_back(idx);
        }
    }
    std::sort(cpu7_vector_.begin(), cpu7_vector_.end());
    cpu7_vector = cpu7_vector_;

    //    if(diff_time>=0)
    //    {
    ////        double large_rc_need = diff_time;
    ////        cpu7_vector+
    //    }
    //    else{
    ////        cpu7_vector-
    //    }
    //    cpu7_vector.push_back(2);
    //    cpu7_vector.push_back(5);
    //    cpu7_vector.push_back(110);
    return;
}


void get_queue_init(){
    std::vector<int> qs[CPU_NUMS];
    std::vector<double> ts[CPU_NUMS];
    double sums[CPU_NUMS];
    for(int i=2; i<rc_times.size(); i++){
        if(std::count(cpu7_vector.begin(), cpu7_vector.end(), i))
            continue;

        for (int j=0; j<CPU_NUMS; j++)
        {
            sums[j] = SumVector(ts[j]);
        }

        int minPosition = std::min_element(sums,sums+4) - sums;
        qs[minPosition].push_back(i);
        ts[minPosition].push_back(rc_times[i]);

    }

    printf("SUM  ");
    for (int j=0; j<CPU_NUMS; j++)
    {
        printf("%f  ", sums[j]);
        cpu_Vectors[j]= qs[j];
    }
    printf("\n");

    return;
}
void get_queue()
{
    std::vector<double> ts[CPU_NUMS];
    double sums[CPU_NUMS];
    //    std::vector<double> t1, t2, t3, t0;
    for (int j=0; j<CPU_NUMS; j++)
    {
        for (int i : cpu_Vectors[j])
        {
            ts[j].push_back(rc_times[i]);
        }
    }

    for (int j=0; j<CPU_NUMS; j++)
    {
        sums[j] = SumVector(ts[j]);
    }

    printf("SUM  ");
    for (int j=0; j<CPU_NUMS; j++)
    {
        printf("%f  ", sums[j]);
    }
    printf("\n");
    //    double sum[]={sum0, sum1, sum2, sum3};
    int minPosition = std::min_element(sums,sums+4) - sums;
    int maxPosition = std::max_element(sums,sums+4) - sums;
    std::vector<int> max_vector = cpu_Vectors[maxPosition];
    std::vector<int> min_vector = cpu_Vectors[minPosition];

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

    cpu_Vectors[maxPosition] = max_vector_;
    cpu_Vectors[minPosition] = min_vector;
    for (int j=0; j<CPU_NUMS; j++)
    {
        std::sort(cpu_Vectors[j].begin(), cpu_Vectors[j].end());
    }


    std::vector<double> t_s[CPU_NUMS];
    for (int j=0; j<CPU_NUMS; j++)
    {
        for (int i : cpu_Vectors[j])
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
        printf("int cpu%d_list[]={", j);
        for (int i = 0; i < cpu_Vectors[j].size(); ++i)
        {
            printf("%d,", cpu_Vectors[j][i]);
        }
        printf("};\n");
    }
    printf("int cpu7_list[]={");
    for (int i = 0; i < cpu7_vector.size(); ++i)
    {
        printf("%d,", cpu7_vector[i]);
    }
    printf("};\n");


    printf("SUM  ");
    for (int j=0; j<CPU_NUMS; j++)
    {
        printf("%f  ", sums[j]);
    }
    printf("\n");
    //    printf("SUM  %f, %f, %f, %f\n", sum0, sum1, sum2, sum3);
}
void WriteBinaryFile( const char* comment)
{
    char path[256];
    sprintf(path, MODEL_DIR "%s.dat", comment);
    std::ofstream osData(path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

    for (int i=0; i<CPU_NUMS; i++)
    {
        int n = CPU_NUMS - i - 1;
        int cpuListSize = cpu_Vectors[n].size();
        osData.write(reinterpret_cast<char *>(&cpuListSize), sizeof(cpuListSize));
        osData.write(reinterpret_cast<char *>(cpu_Vectors[n].data()), cpuListSize*sizeof(cpu_Vectors[n].front()) );

    }
    int cpu7ListSize = cpu7_vector.size();
    osData.write(reinterpret_cast<char *>(&cpu7ListSize), sizeof(cpu7ListSize));
    osData.write(reinterpret_cast<char *>(cpu7_vector.data()), cpu7ListSize*sizeof(cpu7_vector.front()) );
    osData.close();
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

    for(int l=0; l<20; l++)
    {
        //        sleep(1);
        double sum_time = 0;

        get_time(model_name, in, opt);

        printf("rc_times_len=%zu\n", rc_times.size());
        for (int i = 0; i < rc_times.size(); i++)
        {
            //        printf("%f, ", rc_times[i]);
            sum_time += rc_times[i];
        }
        loop_sum = sum_time;
        printf("sum = %f\n", sum_time);
        get_largecore_queue(l);
        get_queue_init();
        //        printf("==========================-------------------------------============================================================%d\n\n", finish_f);
        if(finish_f)
            break;

        //    on_read("MobileNet", ncnn::Mat(227, 227, 3), opt);
        //    on_pipe("GoogleNet", ncnn::Mat(224, 224, 3), opt);
        //    on_infer("MobileNet", ncnn::Mat(227, 227, 3), opt);

        //    benchmark("AlexNet", ncnn::Mat(227, 227, 3), opt);
        // run
        //        benchmark_new("AlexNet", ncnn::Mat(227, 227, 3), opt);
        //        benchmark_new("GoogleNet", ncnn::Mat(227, 227, 3), opt);
        for (int i = 0; i < 10; i++)
        {
            //            for (int j=0; j<CPU_NUMS; j++)
            //            {
            //                printf("int cpu%d_list[]={", j);
            //                for (int i = 0; i < cpu_Vectors[j].size(); ++i)
            //                {
            //                    printf("%d,", cpu_Vectors[j][i]);
            //                }
            //                printf("};\n");
            //            }
            //            printf("int cpu7_list[]={");
            //            for (int i = 0; i < cpu7_vector.size(); ++i)
            //            {
            //                printf("%d,", cpu7_vector[i]);
            //            }
            //            printf("};\n");
            //                    sleep(1);
            sum_time = 0;

            benchmark_new(model_name, in, opt);

            printf("rc_times_len=%zu\n", rc_times.size());
            for (int i = 0; i < rc_times.size(); i++)
            {
                //            printf("%f, ", rc_times[i]);
                sum_time += rc_times[i];
            }
            printf("sum = %f\n", sum_time);
            loop_sum = sum_time;
            get_queue();
        }

        //        printf("int cpu0_list[]={");
        //        for (int i = 0; i < cpu0_vector.size(); ++i)
        //        {
        //            printf("%d,", cpu0_vector[i]);
        //        }
        //        printf("};\n");
        //        printf("int cpu1_list[]={");
        //        for (int i = 0; i < cpu1_vector.size(); ++i)
        //        {
        //            printf("%d,", cpu1_vector[i]);
        //        }
        //        printf("};\n");
        //        printf("int cpu2_list[]={");
        //        for (int i = 0; i < cpu2_vector.size(); ++i)
        //        {
        //            printf("%d,", cpu2_vector[i]);
        //        }
        //        printf("};\n");
        //        printf("int cpu3_list[]={");
        //        for (int i = 0; i < cpu3_vector.size(); ++i)
        //        {
        //            printf("%d,", cpu3_vector[i]);
        //        }
        //        printf("};\n");
        //        printf("int cpu7_list[]={");
        //        for (int i = 0; i < cpu7_vector.size(); ++i)
        //        {
        //            printf("%d,", cpu7_vector[i]);
        //        }
        //        printf("};\n");

        printf("\n===================\n================\n==============\n\n");

        last_loop_sum = loop_sum;
        last_skp_time = for_skp_time;
        last_cal_time = for_cal_time;
        last_largecore_rc_time = largecore_rc_time;
        for (int i=0; i<CPU_NUMS; i++)
        {
            last_cpu_times[i] = cpu_times[i];
        }
        //        last_cpu3_time = cpu3_time;
        //        last_cpu2_time = cpu2_time;
        //        last_cpu1_time = cpu1_time;
        //        last_cpu0_time = cpu0_time;
        WriteBinaryFile(model_name);

    }
    //    WriteBinaryFile(model_name);
    //    ReadBinaryFile(model_name);

    //        benchmark_new("MobileNet", ncnn::Mat(224, 224, 3), opt);
    //        benchmark_new("MobileNetV2", ncnn::Mat(224, 224, 3), opt);
    //        benchmark_new("resnet18", ncnn::Mat(224, 224, 3), opt);

    //    benchmark_new("mobilenet_int8", ncnn::Mat(224, 224, 3), opt);
    //    benchmark_new("googlenet_int8", ncnn::Mat(224, 224, 3), opt);
    //    benchmark_new("resnet18_int8", ncnn::Mat(224, 224, 3), opt);
#if NCNN_VULKAN
    delete g_blob_vkallocator;
    delete g_staging_vkallocator;
#endif // NCNN_VULKAN

    //    printf("end %f\n", ncnn::get_current_time()-start_t);
    return 0;
}
