#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
//#include <unistd.h>
#include <mutex>

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"

bool gRun;
bool SAVE_RESULT = false;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[]) {

    std::cout<<"detection\n";
    signal(SIGINT, sig_handler);

    int image_counter = 0;
    std::string result = "";


    std::string net = "yolo4tiny_fp32.rt";

    std::string root_dir = "yolo4tiny_custom";

    if(argc > 1)
        net = argv[1];
    #ifdef __linux__
        std::string input = "../demo/yolo_test.mp4";
    #elif _WIN32
        std::string input = "..\\..\\..\\demo\\yolo_test.mp4";
    #endif

    if(argc > 2)
        input = argv[2];
    char ntype = 'y';
    if(argc > 3)
        ntype = argv[3][0];
    int n_classes = 80;
    if(argc > 4)
        n_classes = atoi(argv[4]);
    int n_batch = 1;
    if(argc > 5)
        n_batch = atoi(argv[5]);
    bool show = true;
    if(argc > 6)
        show = atoi(argv[6]);
    float conf_thresh=0.3;
    if(argc > 7)
        conf_thresh = atof(argv[7]);

    if(n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");

    if(!show)
        SAVE_RESULT = true;

    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;
    tk::dnn::MobilenetDetection mbnet;

    tk::dnn::DetectionNN *detNN;

    switch(ntype)
    {
        case 'y':
            detNN = &yolo;
            break;
        case 'c':
            detNN = &cnet;
            break;
        case 'm':
            detNN = &mbnet;
            n_classes++;
            break;
        default:
        FatalError("Network type not allowed (3rd parameter)\n");
    }

    detNN->init(net, n_classes, n_batch, conf_thresh);

    gRun = true;

    cv::VideoCapture cap(input);
    if(!cap.isOpened())
        gRun = false;
    else
        std::cout<<"camera started\n";

    std::cout<<"[class probability x_left_top y_left_top x_right_bottom y_right_bottom]\n";

    cv::VideoWriter resultVideo;
    if(SAVE_RESULT) {
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        resultVideo.open("result.mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(w, h));
    }

    cv::Mat frame;
    if(show)
        cv::namedWindow("detection", cv::WINDOW_NORMAL);

    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;
    std::vector<std::string> bboxes_result;

    while(gRun) {
        batch_dnn_input.clear();
        batch_frame.clear();

        for(int bi=0; bi< n_batch; ++bi){
            cap >> frame;
            if(!frame.data)
                break;

            batch_frame.push_back(frame);

            // this will be resized to the net format
            batch_dnn_input.push_back(frame.clone());
        }
        if(!frame.data)
            break;


        bboxes_result.clear();

        //inference
        detNN->update(batch_dnn_input, n_batch);
        if (detNN->detected.size() > 0)
        {
            // result = "ddsfsdf";
            // cl_ = detNN->detected[0].cl
            // prob = detNN->detected[0].prob
            // x = detNN->detected[0].x
            // y = detNN->detected[0].y
            // w = detNN->detected[0].w
            // h = detNN->detected[0].h
            for (int k=0; k<detNN->detected.size(); ++k)
            {
                result = std::to_string(detNN->detected[k].cl) + " " +
                    std::to_string(detNN->detected[k].prob) + " " +
                    std::to_string(detNN->detected[k].x) + " " +
                    std::to_string(detNN->detected[k].y) + " " +
                    std::to_string(detNN->detected[k].x + detNN->detected[k].w) + " " +
                    std::to_string(detNN->detected[k].y + detNN->detected[k].h);
                bboxes_result.push_back(result);
            }


            // std::cout << result;
        }



        std::ofstream outFile(root_dir + "/inference_after/" + std::to_string(image_counter) + ".txt");
        // the important part
        for (const auto &e : bboxes_result) outFile << e << "\n";

        detNN->draw(batch_frame);


        if(show){
            for(int bi=0; bi < n_batch; ++bi){
                cv::imshow("detection", batch_frame[bi]);
                cv::waitKey(1);
                // save images
                cv::imwrite(root_dir + "/inference_after/" + std::to_string(image_counter) + ".jpg", batch_frame[bi]);
                image_counter++;
            }
        }
        if(n_batch == 1 && SAVE_RESULT)
            resultVideo << frame;
    }

    std::cout<<"detection end\n";
    double mean = 0;

    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";
    for(int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean/n_batch<<" ms\t"<<1000/(mean/n_batch)<<" FPS\n"<<COL_END;


    return 0;
}

