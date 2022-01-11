#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
//#include <unistd.h>
#include <mutex>

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"


#include <iostream>
#include <vector>
#include <string>
#include <experimental/filesystem> // http://en.cppreference.com/w/cpp/experimental/fs

std::vector<std::string> get_filenames( std::experimental::filesystem::path path )
{
    namespace stdfs = std::experimental::filesystem ;

    std::vector<std::string> filenames ;

    // http://en.cppreference.com/w/cpp/experimental/fs/directory_iterator
    const stdfs::directory_iterator end{} ;

    for( stdfs::directory_iterator iter{path} ; iter != end ; ++iter )
    {
        // http://en.cppreference.com/w/cpp/experimental/fs/is_regular_file
        if( stdfs::is_regular_file(*iter) ) // comment out if all names (names of directories tc.) are required
            filenames.push_back( iter->path().string() ) ;
    }

    return filenames ;
}


void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
}

int main(int argc, char *argv[]) {

    std::cout<<"detection\n";
    signal(SIGINT, sig_handler);

    int image_counter = 0;
    std::string result = "";


    std::string net = "yolo4tiny_fp32.rt";

    std::string root_dir = "";

    if(argc > 1)
        net = argv[1];

    if(argc > 2)
        root_dir = argv[2];

    #ifdef __linux__
        std::string separator = "/";
    #elif _WIN32
        std::string separator = "\\";
    #endif

    char ntype = 'y';
    if(argc > 3)
        ntype = argv[3][0];

    int n_classes = 11;
    if(argc > 4)
        n_classes = atoi(argv[4]);

    int n_batch = 2;
    // if(argc > 5)
    //     n_batch = atoi(argv[5]);
    // bool show = true;
    // if(argc > 6)
    //     show = atoi(argv[6]);
    float conf_thresh=0.3;
    if(argc > 5)
        conf_thresh = atof(argv[5]);

    if(n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");


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

    std::cout<<"output bbox format: [class probability x_left_top y_left_top x_right_bottom y_right_bottom]\n";

    cv::Mat frame;
    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;
    std::vector<std::string> bboxes_result;

    std::vector<std::string> filenames = get_filenames(root_dir +  separator + "inference_before");

    int index = 0;
    // int image_counter = 0;
    while (index < filenames.size())
    {
        batch_dnn_input.clear();
        batch_frame.clear();

        for(int bi=0; bi< n_batch; ++bi){
            frame = cv::imread(filenames[index]);
            // cap >> frame;
            index++;

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
        if (detNN->detected.size() > 0) {
            for (int k=0; k<detNN->detected.size(); ++k) {
                result = std::to_string(detNN->detected[k].cl) + " " +
                    std::to_string(detNN->detected[k].prob) + " " +
                    std::to_string(detNN->detected[k].x) + " " +
                    std::to_string(detNN->detected[k].y) + " " +
                    std::to_string(detNN->detected[k].x + detNN->detected[k].w) + " " +
                    std::to_string(detNN->detected[k].y + detNN->detected[k].h);
                bboxes_result.push_back(result);
            }
        }

        std::ofstream outFile(root_dir + separator + "inference_after" + separator + std::to_string(image_counter) + ".txt");
        for (const auto &e : bboxes_result) outFile << e << "\n";

        detNN->draw(batch_frame);

        for(int bi=0; bi < n_batch; ++bi){
            cv::imwrite(root_dir + separator + "inference_after" + separator + std::to_string(image_counter) + ".jpg", batch_frame[bi]);
            image_counter++;
        }

        std::cout << "processing image " + std::to_string(image_counter) + '\n';

    }

    // for (const auto& name : get_filenames(root_dir +  separator + "inference_before")) {
    //     frame = cv::imread(name);
    //     batch_dnn_input.clear();
    //     batch_frame.clear();

    //     for(int bi=0; bi< n_batch; ++bi){
    //         cap >> frame;
    //         if(!frame.data)
    //             break;

    //         batch_frame.push_back(frame);

    //         // this will be resized to the net format
    //         batch_dnn_input.push_back(frame.clone());
    //     }
    //     if(!frame.data)
    //         break;

    //     bboxes_result.clear();

    //     //inference
    //     detNN->update(batch_dnn_input, n_batch);
    //     if (detNN->detected.size() > 0) {
    //         for (int k=0; k<detNN->detected.size(); ++k) {
    //             result = std::to_string(detNN->detected[k].cl) + " " +
    //                 std::to_string(detNN->detected[k].prob) + " " +
    //                 std::to_string(detNN->detected[k].x) + " " +
    //                 std::to_string(detNN->detected[k].y) + " " +
    //                 std::to_string(detNN->detected[k].x + detNN->detected[k].w) + " " +
    //                 std::to_string(detNN->detected[k].y + detNN->detected[k].h);
    //             bboxes_result.push_back(result);
    //         }

    //         // std::cout << result;
    //     }

    //     std::ofstream outFile(root_dir + separator + "inference_after" + separator + std::to_string(image_counter) + ".txt");
    //     for (const auto &e : bboxes_result) outFile << e << "\n";

    //     detNN->draw(batch_frame);

    //     // for(int bi=0; bi < n_batch; ++bi){
    //     cv::imwrite(root_dir + separator + "inference_after" + separator + std::to_string(image_counter) + ".jpg", batch_frame[0]);
    //     image_counter++;
    //     // }

    //     std::cout << "processing image " + std::to_string(image_counter) + '\n';
    // }

    std::cout<<"detection end\n";
    double mean = 0;

    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";
    for(int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean/n_batch<<" ms\t"<<1000/(mean/n_batch)<<" FPS\n"<<COL_END;


    return 0;
}

