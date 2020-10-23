//
// Created by himalaya on 3/16/20 at 1:42 PM.
//
#include "algorithms.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>

using namespace std;
using namespace cv;


void PnpBase::setRandomGeneratorSeed(int seed) {
    this->generator.seed(seed);
}

void PnpBase::setUniformDistribution(dtype lower, dtype upper) {
    uniform_distribution = uniform_real_distribution<double>(lower, upper);
}

void PnpBase::setBinomialDistribution(int upper, dtype p) {
    int_noise_distribution = binomial_distribution<int>(upper, p);
}

dtype PnpBase::genNumFromDistribution(int method) {
    switch (method) {
        case 0:
            return uniform_distribution(generator);
        case 1:
            return int_noise_distribution(generator);
        default:
            return -1;
    }
}

void PnpBase::genRect3dPair2d(const ImageBase *cam, std::vector<std::vector<Vec3>> &obj_points, std::vector<std::vector<Vec2>> &img_points, int num_sets, int num_pairs) {
    Mat3 K = cam->getIntrinsic();
    Mat3 R = cam->getRotation();
    Vec3 t = cam->getTranslation();
    obj_points.clear();
    img_points.clear();
    obj_points.resize(num_sets);
    img_points.resize(num_sets);
    int count = 0;
    int margin = 10;
    while (count < num_sets) {
        auto val = genNumFromDistribution(0);
        bool valid = true;
        for (int i = -1; i <= 1; i += 2) {
            for (int j = (i == -1 ? -1 : 1); i == -1 ? j <= 1 : j >= -1; j += (i == -1 ? 2 : -2)) {
                obj_points[count].emplace_back(Vec3(i * val, j * val, 0));
                Vec3 projection = K * (R * obj_points[count].back() + t);
                projection /= projection[2];
                if (projection[0] >= cam->getWidth() - margin || projection[0] < margin ||
                    projection[1] >= cam->getHeight() - margin || projection[1] < margin) {
                    valid = false;
                }
                img_points[count].emplace_back(Vec2(projection[0], projection[1]));
            }
        }
        if (!valid) {
            img_points[count].clear();
            count--;
        }
        count++;
    }

    // verify
//    for (int i = 0; i < num_sets; i++) {
//        for (int j = 0; j < 4; j++)
//            printf("%.4f\t%.4f\t%.4f\n%.4f\t%.4f\n\n", obj_points[i][j][0], obj_points[i][j][1], obj_points[i][j][2], img_points[i][j][0], img_points[i][j][1]);
//    }
}

std::vector<Vec3> PnpBase::loadObjPointsFromFile(const std::string &file_path) {
    ifstream fin(file_path, ifstream::in);
    if (!fin.is_open()) {
        cerr << "Object points file not existing or invalid." << endl;
        exit(EXIT_FAILURE);
    }
    vector<Vec3> obj_points;
    string line;
    while (getline(fin, line)) {
        if (!line.empty()) {
            stringstream ss(line);
            Vec3 point;
            ss >> point[0] >> point[1] >> point[2];
            obj_points.emplace_back(point);
        }
    }
    fin.close();
    // verify
//     for (const Vec3 &obj: obj_points) {
//         cout << obj << endl;
//     }
    obj_points.shrink_to_fit();
    return obj_points;
}
std::vector<Vec2> PnpBase::loadImgPointsFromFile(const std::string &file_path) {
    ifstream fin(file_path, ifstream::in);
    if (!fin.is_open()) {
        cerr << "Image points file not existing or invalid." << endl;
        exit(EXIT_FAILURE);
    }
    vector<Vec2> img_points;
    string line;
    while (getline(fin, line)) {
        if (!line.empty()) {
            stringstream ss(line);
            Vec2 point;
            ss >> point[0];
            ss >> point[1];
            img_points.emplace_back(point);
        }
    }
    fin.close();
    // verify
//    for (const Vec2 &img_point: img_points) {
//        cout << img_point << endl;
//    }
    img_points.shrink_to_fit();
    return img_points;
}

struct CallbackParams {
    vector<Vec2> img_points;
    cv::Mat dst;
    std::string window_name;
    const cv::Mat &src;
    CallbackParams(const cv::Mat &src__img): src(src__img) {}
};
/**
 * @brief OpenCV mouse event interrupt.
 * @param event Event id.
 * @param x Mouse x coordinate.
 * @param y Mouse y coordinate.
 * @param flag Specific event. (refer to docs)
 * @param params User defined parameters.
 */
static void onMouse(int event, int x, int y, int flag, void *params) {
    auto parameters = (CallbackParams *)params;
    auto &img_points = parameters->img_points;
    auto dst = parameters->dst;
    auto src = parameters->src;
    auto window_name = parameters->window_name;
    if (event == EVENT_LBUTTONDOWN) {
        if (img_points.size() >= 4) {
            cout << "Cannot select more than 4 points. Press Enter or Right Click to undo" << "\n";
            return;
        }
        img_points.emplace_back(Vec2(x, y));
        src.copyTo(dst);
        for (const auto &img_point: img_points) {
            circle(dst, Point((int) img_point[0], (int) img_point[1]), 5, Scalar(0, 0, 255), FILLED);
        }
        imshow(window_name, dst);
    } else if (event == EVENT_RBUTTONDOWN) {
        if (!img_points.empty()) img_points.pop_back();
        src.copyTo(dst);
        for (const auto &img_point: img_points) {
            circle(dst, Point((int) img_point[0], (int) img_point[1]), 5, Scalar(0, 0, 255), FILLED);
        }
        imshow(window_name, dst);
    }
}

std::vector<Vec2> PnpBase::selectImgPointsFromImage(const ImageBase *cam) {
    CallbackParams params(cam->getImage());
    params.window_name = "Rectangle Vertices Selection";
    namedWindow(params.window_name, WINDOW_GUI_EXPANDED | WINDOW_NORMAL | WINDOW_KEEPRATIO);
    resizeWindow(params.window_name, 1024, 576);
    setMouseCallback(params.window_name, onMouse, (void *)&params);
    imshow(params.window_name, params.src);
    while (1) {
        // press Enter to finish
        if (waitKey(10) == 13) break;
    }
    destroyAllWindows();
    return params.img_points;
}

bool PnpBase::saveImgPoints(const std::string &file_path, const std::vector<Vec2> &img_points) {
    ofstream fout(file_path, ofstream::out);
    if (img_points.size() != 4) {
        cerr << "Select 4 image points first." << endl;
        exit(EXIT_FAILURE);
        return false;
    }
    for (const auto &img_point: img_points) {
        fout << img_point[0] << "\t" << img_point[1] << "\n";
    }
    return true;
}

double PnpBase::reprojectError(const ImageBase *cam, const std::vector<Vec3> &obj_points, const std::vector<Vec2> &img_points) {
    double error = 0;
    int num_points = obj_points.size();
    Mat3 K = cam->getIntrinsic();
    Mat3 R = cam->getRotation();
    Vec3 t = cam->getTranslation();
    for (int i = 0; i < num_points; i++) {
        Vec3 x_prime = K * (R * obj_points[i] + t);
        x_prime /= x_prime[2];
        // cout << "Point " << i << ": " << "\n";
         printf("%.5f \t %.5f\n%.5f \t %.5f\n", img_points[i][0], x_prime[0], img_points[i][1], x_prime[1]);
        error += sqrt(pow((img_points[i][0]) - (x_prime[0]), 2) + pow((img_points[i][1]) - (x_prime[1]), 2));
    }
    error /= num_points;
    cout << error << endl;
    return error;
}


void P3P::solve(ImageBase *cam, const vector<Vec3> &obj_points, const vector<Vec2> &img_points, int method) {
    switch (method) {
        case 0:
            P3P_LKneip(cam, obj_points, img_points);
            break;
        case 1:
            break;
        default:
            break;
    }
}

void P3P::P3P_LKneip(ImageBase *cam, const vector<Vec3> &obj_points, const vector<Vec2> &img_points) {
    if (obj_points.size() < 4 || img_points.size() < 4) {
        throw std::runtime_error("Cannot uniquely recover extrinsics with less than 4 points.");
    }
    vector<Vec3> obj_points_3{obj_points[0], obj_points[1], obj_points[2]};
    vector<Vec2> img_points_2{img_points[0], img_points[1], img_points[2]};
    auto candidates = P3P_LKneipCandidates(cam, obj_points_3, img_points_2);
    Mat3 K = cam->getIntrinsic();
    
}

std::vector<Eigen::Matrix<dtype, 3, 4>>
P3P::P3P_LKneipCandidates(const ImageBase *cam, const vector<Vec3> &obj_points, const vector<Vec2> &img_points) {
    std::vector<Eigen::Matrix<dtype, 3, 4>> ans;
    Mat3 K = cam->getIntrinsic();
    Mat3 K_inv;
    K_inv << 1. / K(0, 0), 0, -K(0, 2) / K(0, 0),
            0, 1. / K(1, 1), -K(1, 2) / K(1, 1),
            0, 0, 1;
    // image points in camera coordinate system
    vector<Vec3> img_points_cam(3);
    for (int i = 0; i < img_points_cam.size(); i++) {
        const auto &x = img_points[i][0];
        const auto &y = img_points[i][1];
        img_points_cam[i] << K_inv(0, 0) * x + K_inv(0, 2),
                             K_inv(1, 1) * y + K_inv(1, 2),
                             1;
    }
//    for (const auto &e: img_points_cam) {
//        cout << e << "\n" << endl;
//    }
    // transform camera frame into tao frame.
    Mat3 rot_tao;
    rot_tao.col(0) = img_points_cam[0].normalized();
    rot_tao.col(2) = img_points_cam[0].cross(img_points_cam[1]).normalized();
    rot_tao.col(1) = rot_tao.col(2).cross(rot_tao.col(0)).normalized();
    rot_tao.transposeInPlace();

    // transform inertial frame into eta frame
    Mat3 rot_eta;
    Vec3 t_eta;
    rot_eta.col(0) = (obj_points[1] - obj_points[0]).normalized();
    rot_eta.col(2) = rot_eta.col(0).cross(obj_points[2] - obj_points[0]).normalized();
    rot_eta.col(1) = rot_eta.col(2).cross(rot_eta.col(0)).normalized();
    rot_eta.transposeInPlace();
    t_eta = -rot_eta * obj_points[0];

    Vec3 f3_tao = rot_tao * img_points_cam[2].normalized();
    dtype phi1 = f3_tao[0] / f3_tao[2];
    dtype phi2 = f3_tao[1] / f3_tao[2];

    // P3 in frame eta
    Vec3 P3_eta = rot_eta * obj_points[2] + t_eta;
    // cout << P3_eta << endl;

    // compute all the coefficients
    const dtype &p1 = P3_eta[0], &p2 = P3_eta[1];
    dtype d12 = (obj_points[0] - obj_points[1]).norm(); // Euclidean distance of P_1P_2
    dtype d12_2 = d12 * d12;
    dtype phi1_2 = phi1 * phi1, phi2_2 = phi2 * phi2;
    dtype p1_2 = p1 * p1, p2_2 = p2 * p2;
    dtype p1_3 = p1_2 * p1, p2_3 = p2_2 * p2;
    dtype p1_4 = pow(p1, 4), p2_4 = pow(p2, 4);
    // compute cot(beta)
    dtype cos_beta = img_points_cam[0].normalized().dot(img_points_cam[1].normalized());
    dtype b = (cos_beta > 0 ? 1 : -1) * sqrt(1 / (1 - cos_beta * cos_beta) - 1);
    dtype b_2 = b * b;

    dtype a4 = -phi2_2 * p2_4 - phi1_2 * p2_4 - p2_4;
    dtype a3 = 2 * p2_3 * d12 * b + 2 * phi2_2 * p2_3 * d12 * b - 2 * phi1 * phi2 * p2_3 * d12;
    dtype a2 = -phi2_2 * p1_2 * p2_2 - phi2_2 * p2_2 * d12_2 * b_2 - phi2_2 * p2_2 * d12_2 + phi2_2 * p2_4
               + phi1_2 * p2_4 + 2 * p1 * p2_2 * d12 + 2 * phi1 * phi2 * p1 *p2_2 * d12 * b
               - phi1_2 * p1_2 * p2_2 + 2 * phi2_2 * p1 * p2_2 * d12 - p2_2 * d12_2 * b_2 - 2 * p1_2 * p2_2;
    dtype a1 = 2 * p1_2 * p2 * d12 * b + 2 * phi1 * phi2 * p2_3 * d12
               - 2 * phi2_2 * p2_3 * d12 * b - 2 * p1 * p2 * d12_2 * b;
    dtype a0 = -2 * phi1 * phi2 * p1 * p2_2 * d12 * b + phi2_2 * p2_2 * d12_2 + 2 * p1_3 * d12
               - p1_2 * d12_2 + phi2_2 * p1_2 * p2_2 - p1_4 - 2 * phi2_2 * p1 * p2_2 * d12
               + phi1_2 * p1_2 * p2_2 + phi2_2 * p2_2 * d12_2 * b_2;
    auto cos_theta_candidates = solveQuartic(a4, a3, a2, a1, a0);
    // for (const auto &e: cos_theta_candidates) {
    //     cout << e << endl;
    // }
    vector<dtype> cot_alpha_candidates;
    for (const auto &e: cos_theta_candidates) {
        dtype phi1_over_phi2 = phi1 / phi2;
        cot_alpha_candidates.emplace_back((phi1_over_phi2 * p1 + e * p2 - d12 * b)
                                         / (phi1_over_phi2 * e * p2 - p1 + d12));
    }

    // size is up to 4
    for (int i = 0; i < cos_theta_candidates.size(); i++) {
        const auto &cos_theta = cos_theta_candidates[i];
        const auto &cot_alpha = cot_alpha_candidates[i];
        dtype sin_theta, sin_alpha, cos_alpha;
        sin_theta = (f3_tao[2] < 0 ? 1 : -1 ) * sqrt(1 - cos_theta * cos_theta);

        // sin(alpha) is always positive since it's an angle of a triangle
        dtype cot_alpha_2 = cot_alpha * cot_alpha;
        sin_alpha = sqrt(1 / (1 + cot_alpha_2));
        cos_alpha = (cot_alpha > 0 ? 1 : -1) * sqrt(cot_alpha_2 / (1 + cot_alpha_2));

        dtype temp = sin_alpha * b + cos_alpha;
        Vec3 C_eta(d12 * cos_alpha * temp,
                   d12 * sin_alpha * cos_theta * temp,
                   d12 * sin_alpha * sin_theta * temp);

        Mat3 Q;
        Q << -cos_alpha,        -sin_alpha * cos_theta,         -sin_alpha * sin_theta,
              sin_alpha,        -cos_alpha * cos_theta,         -cos_alpha * sin_theta,
              0        ,        -sin_theta            ,         cos_theta;

        // camera center coordinates
        Vec3 C = obj_points[0] + rot_eta.transpose() * C_eta;
        Mat3 R = rot_tao.transpose() * Q * rot_eta;
        Vec3 t = -R * C;
        Eigen::Matrix<dtype, 3, 4> pose;
        pose << R, t;
        ans.emplace_back(pose);
         cout << pose << endl;
    }
    return ans;
}

std::vector<dtype> P3P::solveQuartic(const dtype &a4, const dtype &a3, const dtype &a2, const dtype &a1, const dtype &a0) {
    std::vector<dtype> ans;
    if (abs(a4) < 1e-8) {
        throw std::runtime_error("Coefficient of quartic term is zero");
    }
    dtype b = a3 / a4;

    dtype a4_2 = a4 * a4;
    dtype a4_3 = a4_2 * a4;
    dtype a3_2 = a3 * a3;
    // convert the equation into depressed form y^4 + py^2 + qy + r = 0
    // ref: quartic equation in Wikipeida
    dtype alpha = (8 * a2 * a4 - 3 * a3_2) / 8 / a4_2;
    dtype beta = (pow(a3, 3) - 4 * a2 * a3 * a4 + 8 * a1 * a4_2) / 8 / a4_3;
    dtype gamma = (-3 * a3_2 * a3_2 + 256 * a0 * a4_3 - 64 * a1 * a3 * a4_2 + 16 * a2 * a3_2 * a4)
              / 256 / (a4_2 * a4_2);
    
    dtype alpha_2 = alpha * alpha;
    dtype P = -alpha_2 / 12 - gamma;
    dtype Q = -alpha_2 * alpha / 108 + alpha * gamma / 3 - beta * beta / 8;
    std::complex<dtype> R = -Q / 2 + sqrt(Q * Q / 4 + pow(P, 3) / 27);
    std::complex<dtype> U = pow(R, (dtype)(1.0 / 3));
    
    std::complex<dtype> y;
    if (U.real() == 0) {
        y = -5. * alpha / 6 - pow(Q, 1. / 3);
    }
    else {
        y = -(dtype)5. * alpha / 6 + U - P / 3 / U;
    }
    std::complex<dtype> W = sqrt(alpha + y * (dtype)2);

    std::complex<dtype> first_item = -a3 / 4 / a4;
    std::complex<dtype> sqrt_aux1 = 3 * alpha + (dtype)2 * y;
    std::complex<dtype> sqrt_aux2 = 2 * beta / W;
    std::complex<dtype> second_item(0);
    for (int i = -1; i <= 1; i += 2) {
        for (int j = -1; j <= 1; j += 2) {
            second_item += i == -1 ? -W : W;
            std::complex<dtype> sqrt_item(0);
            sqrt_item += -sqrt_aux1;
            sqrt_item += i == -1 ? sqrt_aux2 : -sqrt_aux2;
            sqrt_item = sqrt(sqrt_item);
            second_item += j == -1 ? -sqrt_item : sqrt_item;
            second_item /= (dtype)2.;
            std::complex<dtype> sub_ans = first_item + second_item;
            // keep reasonable solutions. if the image part is small, we regards it as the result of noise and keep the real part.
            if (abs(sub_ans.imag()) < 1e-5) {
                ans.emplace_back(sub_ans.real());
            }
            second_item = 0.;
        }
    }
    // second_item = (W + sqrt(-(sqrt_aux1 + sqrt_aux2))) / (dtype)2;
    // ans.emplace_back((first_item + second_item).real());
    // second_item = (W - sqrt(-(sqrt_aux1 + sqrt_aux2))) / (dtype)2;
    // ans.emplace_back((first_item + second_item).real());
    // second_item = (-W + sqrt(-(sqrt_aux1 - sqrt_aux2))) / (dtype)2;
    // ans.emplace_back((first_item + second_item).real());
    // second_item = (-W - sqrt(-(sqrt_aux1 - sqrt_aux2))) / (dtype)2;
    // ans.emplace_back((first_item + second_item).real());
    return ans;
}


