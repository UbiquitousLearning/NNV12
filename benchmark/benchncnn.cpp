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

char globalbinpath[256];

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

    int open = net.load_model(binpath);
    if (open<0)
    {
        DataReaderFromEmpty dr;
        net.load_model(dr);
    }

    //    DataReaderFromEmpty dr;
    //    net.load_model(dr);
    double load_model_end = ncnn::get_current_time();
    double load_model_time = load_model_end - load_model_start;

    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    if (g_enable_cooling_down)
    {
        // sleep 10 seconds for cooling down SOC  :(
#ifdef _WIN32
        Sleep(10 * 1000);
#elif defined(__unix__) || defined(__APPLE__)
//        sleep(10);
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
    for (int i = 0; i < g_warmup_loop_count; i++)
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
            //            float* ptr = (float*)out.data;
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

    fprintf(stderr, "%20s  load param = %7.2f  load model = %7.2f  first = %7.2f  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, load_param_time, load_model_time, first_time, time_min, time_max, time_avg);
}
pthread_t t_infer;
pthread_t t_pipe;
pthread_t t_dr;
//extern double infer_start;
//extern double infer_end;
//extern double infer_time;
//extern double dr_start;
//extern double dr_end;
//extern double dr_time;
//extern double pipe_start;
//extern double pipe_end;
//extern double pipe_time;

//double infer_start;
//double infer_end;
//double infer_time;
//double dr_start;
//double dr_end;
//double dr_time;
//double pipe_start;
//double pipe_end;
//double pipe_time;
size_t dr_cpu=1;//1;
size_t pipe_cpu=6;//6;
size_t infer_cpu=7;//0;
void inference(const ncnn::Net& net, const ncnn::Mat& in){
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

    infer_end = ncnn::get_current_time();
    infer_time = infer_end - infer_start;
    printf("for_skp_time=%f for_cal_time=%f\n", for_skp_time, for_cal_time);
    fprintf(stderr, "infer time =  %7.2f\n", infer_time);

    //    pthread_exit(&t_infer);

}
typedef struct infer_args
{
    ncnn::Net net;
    ncnn::Mat in;
}infer_args;

void *infer_thread(void *args){
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
    auto *stu1=(ncnn::Net*)args;
    printf("inf tid=%ld,cpu=%d\n", pthread_self(), sched_getcpu());
    inference(*stu1, ncnn::Mat(227, 227, 3));
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

    dr_start = ncnn::get_current_time();

    //    printf("%d\n", fork());


    int open = net->load_model_dr(globalbinpath);
    if (open<0)
    {
        printf("load file files\n");
        DataReaderFromEmpty dr;
        net->load_model_dr(dr);
    }


    //    DataReaderFromEmpty dr;
    //    net->load_model_dr(dr);

    dr_end = ncnn::get_current_time();
    dr_time = dr_end - dr_start;

    //    fprintf(stderr, "load model = %7.2f\n", dr_time);
    //    printf("dr fork %d", fork());
    //    while(1)
    //    {
    //        sleep(1);
    //        printf("drr tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    //    }

    //    sleep(200000);
    //    while(1);
    pthread_exit(&t_dr);


    //    inference(*net, ncnn::Mat(227, 227, 3));
    //    fprintf(stderr, "inference\n");
}

void *pipe_thread(void *args){
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
    auto *net=(ncnn::Net*)args;

    //    printf("pip tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    //    pipe_start = ncnn::get_current_time();

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
    pthread_exit(&t_pipe);

    //    inference(*net, ncnn::Mat(227, 227, 3));
    //    fprintf(stderr, "inference\n");
}

void *tmp_thread(void *args){
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
    auto *net=(ncnn::Net*)args;

    //    printf("pip tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    //    pipe_start = ncnn::get_current_time();


    fprintf(stderr, "tmpend\n");
    //    printf("pp fork %d", fork());
    //    while(1)
    //    {
    //        sleep(1);
    //        printf("pip tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    //    }

    //    sleep(200000);
    //    while(1);
    pthread_exit(&t_pipe);

    //    inference(*net, ncnn::Mat(227, 227, 3));
    //    fprintf(stderr, "inference\n");
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

    char binpath[256];
    sprintf(binpath, MODEL_DIR "%s.bin", comment);


    sprintf(globalbinpath, MODEL_DIR "%s.bin", comment);

    double load_param_start = ncnn::get_current_time();
    net.load_param(parampath);
    double load_param_end = ncnn::get_current_time();
    double load_param_time = load_param_end - load_param_start;
    //
    //    double d_start = ncnn::get_current_time();
    //    int open = net.load_model_dr(globalbinpath);
    //    if (open<0)
    //    {
    //        printf("load file files\n");
    //        DataReaderFromEmpty dr;
    //        net.load_model_dr(dr);
    //    }
    ////
    //    double d_end = ncnn::get_current_time();
    //    double d_time = d_end - d_start;
    //    fprintf(stderr, "load model init = %7.2f\n", d_time);
    //    net.load_model_pipe();



#ifdef __CPU_MASK__
    cpu_set_t mask1;  //CPU核的集合
    CPU_ZERO(&mask1);    //置空
    CPU_SET(7,&mask1);   //设置亲和力值
    if (sched_setaffinity(0, sizeof(mask1), &mask1) == -1)//设置线程CPU亲和力
    {
        printf("warning: could not set CPU affinity, continuing...\n");
    }
#endif
    printf("inf tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());

    dr_start = ncnn::get_current_time();

    int open = net.load_model_dr(globalbinpath);
    if (open<0)
    {
        printf("load file files\n");
        DataReaderFromEmpty dr;
        net.load_model_dr(dr);
    }

    dr_end = ncnn::get_current_time();
    dr_time = dr_end - dr_start;
    fprintf(stderr, "load model = %7.2f\n", dr_time);

    pipe_start = ncnn::get_current_time();
    net.load_model_pipe();
    pipe_end = ncnn::get_current_time();
    pipe_time = pipe_end - pipe_start;
    fprintf(stderr, "load pipeline = %7.2f\n", pipe_time);

    //     inference


    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    ncnn::Mat out;

    infer_start = ncnn::get_current_time();

    ncnn::Extractor ex = net.create_extractor();
    ex.input(input_names[0], in);
    ex.extract(output_names[0], out);
    //    float* ptr = (float*)out.data;
    //    printf("%f\n", *ptr);

    infer_end = ncnn::get_current_time();
    infer_time = infer_end - infer_start;
    fprintf(stderr, "infer time =  %7.2f\n", infer_time);
    printf("for_skp_time=%f for_cal_time=%f\n", for_skp_time, for_cal_time);

    double a[3] = {dr_end, pipe_end, infer_end};
    double *b;
    b = std::max_element(a, a+3);
    double max = *b;

    double a1[3] = {dr_start, pipe_start, infer_start};
    double *b1;
    b1 = std::min_element(a1, a1+3);
    double min = *b1;

    double tt_time = max - min;

    fprintf(stderr, "total time = %7.2f\n", tt_time);
    // inference
    for(int ii=0; ii<1;ii++)
    {
        const std::vector<const char*>& input_names__ = net.input_names();
        const std::vector<const char*>& output_names__ = net.output_names();

        ncnn::Mat out__;

        infer_start = ncnn::get_current_time();

        ncnn::Extractor ex__ = net.create_extractor();
        ex__.input(input_names__[0], in);
        ex__.extract(output_names__[0], out__);

        //        float* ptr = (float*)out__.data;
        //        printf("%f\n", *ptr);

        infer_end = ncnn::get_current_time();
        infer_time = infer_end - infer_start;
        fprintf(stderr, "infer time =  %7.2f\n", infer_time);
    }


    fprintf(stderr, "-------\n");
    /*
    //    int open = net.load_model_dr(globalbinpath);
    //    if (open<0)
    //    {
    //        printf("load file files\n");
    //        DataReaderFromEmpty dr;
    //        net.load_model_dr(dr);
    //    }
    //    net.load_model_pipe();
    //    pipe_end = ncnn::get_current_time();
    //    pipe_time = pipe_end - pipe_start;
    //    ncnn::Mat out;
    //    const std::vector<const char*>& input_names = net.input_names();
    //    const std::vector<const char*>& output_names = net.output_names();
    //    ncnn::Extractor ex = net.create_extractor();
    //    ex.input(input_names[0], in);
    //    ex.extract(output_names[0], out);


    //    std::thread t2(tmp_thread, (void *)&net);
    //    std::thread t1(tmp_thread, (void *)&net);

//    std::thread t2(pipe_thread, (void *)&net);
    std::thread t1(dr_thread, (void *)&net);
    //#ifdef __CPU_MASK__
    //    cpu_set_t mask;  //CPU核的集合
    //    CPU_ZERO(&mask);    //置空
    //    CPU_SET(infer_cpu,&mask);   //设置亲和力值
    //    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//设置线程CPU亲和力
    //        {
    //        printf("warning: could not set CPU affinity, continuing...\n");
    //        }
    //#endif
    printf("inf tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    //    sleep(0);
        net.load_model_pipe();
        pipe_end = ncnn::get_current_time();
        pipe_time = pipe_end - pipe_start;

    inference(net, in);


    double a_[3] = {dr_end, pipe_end, infer_end};
    double *b_;
    b_ = std::max_element(a_, a_+3);
    double max_ = *b_;

    double a1_[3] = {dr_start, pipe_start, infer_start};
    double *b1_;
    b1_ = std::min_element(a1_, a1_+3);
    double min_ = *b1_;

    double tt_time_ = max_ - min_;

    fprintf(stderr, "total time = %7.2f\n", tt_time_);
    printf("load start=%7.2f, end=%7.2f, tt=%7.2f\n", dr_start-min_, dr_end-min_, dr_end-dr_start);
    printf("pipe start=%7.2f, end=%7.2f, tt=%7.2f\n", pipe_start-min_, pipe_end-min_, pipe_end-pipe_start);
    printf("infe start=%7.2f, end=%7.2f, tt=%7.2f\n", infer_start-min_, infer_end-min_, infer_end-infer_start);

    // inference
    //    sleep(0);
    for(int ii=0; ii<10;ii++)
    {
        const std::vector<const char*>& input_names__ = net.input_names();
        const std::vector<const char*>& output_names__ = net.output_names();

        ncnn::Mat out__;

        infer_start = ncnn::get_current_time();

        ncnn::Extractor ex__ = net.create_extractor();
        ex__.input(input_names__[0], in);
        ex__.extract(output_names__[0], out__);

        //        float* ptr = (float*)out__.data;
        //        printf("%f\n", *ptr);

        infer_end = ncnn::get_current_time();
        infer_time = infer_end - infer_start;
        fprintf(stderr, "infer time =  %7.2f\n", infer_time);
    }

//    t2.join();
    t1.join();
*/

    /*
    fprintf(stderr, "-------\n");
    std::thread tt(total_thread, (void *)&net);
    tt.join();
*/
    /*
    std::thread t3(infer_thread, (void *)&net);
    std::thread t2(pipe_thread, (void *)&net);

    cpu_set_t mask;  //CPU核的集合
    CPU_ZERO(&mask);    //置空
    CPU_SET(1,&mask);   //设置亲和力值
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)//设置线程CPU亲和力
        {
        printf("warning: could not set CPU affinity, continuing...\n");
        }

    dr_start = ncnn::get_current_time();
    DataReaderFromEmpty dr;
    net.load_model_dr(dr);
    dr_end = ncnn::get_current_time();
    dr_time = dr_end - dr_start;
    fprintf(stderr, "load model = %7.2f\n", dr_time);

//    sleep(200000);
    t3.join();
    t2.join();
*/
    /*
    fprintf(stderr, "-------\n");
    cpu_set_t mask3;  //CPU核的集合
    CPU_ZERO(&mask3);    //置空
    CPU_SET(7,&mask3);   //设置亲和力值
    if (sched_setaffinity(0, sizeof(mask3), &mask3) == -1)//设置线程CPU亲和力
        {
        printf("warning: could not set CPU affinity, continuing...\n");
        }

    dr_start = ncnn::get_current_time();
    DataReaderFromEmpty dr;
    net.load_model_dr(dr);
    dr_end = ncnn::get_current_time();
    dr_time = dr_end - dr_start;
    fprintf(stderr, "load model = %7.2f\n", dr_time);

    pipe_start = ncnn::get_current_time();
    net.load_model_pipe();
    pipe_end = ncnn::get_current_time();
    pipe_time = pipe_end - pipe_start;
    fprintf(stderr, "load pipeline = %7.2f\n", pipe_time);
    inference(net, in);
*/



    //    if (pthread_create(&t_infer, NULL, infer_thread, (void *)&net) != 0) {
    //        fprintf(stderr, "pthread_create() error");
    //    }
    /*
    if (pthread_create(&t_pipe, NULL, pipe_thread, (void *)&net) != 0) {
        fprintf(stderr, "pthread_create() error");
    }
    if (pthread_create(&t_dr, NULL, dr_thread, (void *)&net) != 0) {
        fprintf(stderr, "pthread_create() error");
    }
    inference(net, in);
    int alg=SCHED_RR;
    fprintf(stderr, "%d\n", sched_get_priority_max(alg));
    struct sched_param sp1;
    sp1.sched_priority=sched_get_priority_max(alg);
    pthread_setschedparam(t_dr, alg, &sp1);
    pthread_setschedparam(t_pipe, alg, &sp1);
//    pthread_join(t_infer, NULL);
    pthread_join(t_pipe, NULL);
    pthread_join(t_dr, NULL);
*/



    //    pthread_exit(NULL);






    //    infer_args arg_pack;
    //    arg_pack.net = net;
    //    arg_pack.in = in;
}


int main(int argc, char** argv)
{


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


    benchmark("alexnet", ncnn::Mat(227, 227, 3), opt);
    benchmark("googlenet", ncnn::Mat(227, 227, 3), opt);
    // run
//    benchmark_new("mobilenet_int8", ncnn::Mat(224, 224, 3), opt);
    //        benchmark_new("AlexNet", ncnn::Mat(227, 227, 3), opt);
    //    benchmark_new("GoogleNet", ncnn::Mat(227, 227, 3), opt);
    //        benchmark_new("MobileNet", ncnn::Mat(224, 224, 3), opt);
    //    benchmark_new(MobileNetV2", ncnn::Mat(224, 224, 3), opt);
    //    sleep(50);
    //    benchmark_new("resnet18", ncnn::Mat(224, 224, 3), opt);
    //    sleep(100);

        benchmark("squeezenet", ncnn::Mat(227, 227, 3), opt);
    //
    //    benchmark("squeezenet_int8", ncnn::Mat(227, 227, 3), opt);
    //
        benchmark("mobilenet", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("mobilenet_int8", ncnn::Mat(224, 224, 3), opt);
    //
        benchmark("mobilenet_v2", ncnn::Mat(224, 224, 3), opt);
    //
    //    // benchmark("mobilenet_v2_int8", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("mobilenet_v3", ncnn::Mat(224, 224, 3), opt);
    //
        benchmark("shufflenet", ncnn::Mat(224, 224, 3), opt);
    //
        benchmark("shufflenet_v2", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("mnasnet", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("proxylessnasnet", ncnn::Mat(224, 224, 3), opt);
    //
        benchmark("efficientnet_b0", ncnn::Mat(224, 224, 3), opt);
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
        benchmark("resnet18", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("resnet18_int8", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("alexnet", ncnn::Mat(227, 227, 3), opt);
    //
    //    benchmark("vgg16", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("vgg16_int8", ncnn::Mat(224, 224, 3), opt);
    //
        benchmark("resnet50", ncnn::Mat(224, 224, 3), opt);
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

    return 0;
}
