//
// Created by himalaya on 3/11/20 at 5:39 PM.
//

#ifndef SDF2SDF_ALGORITHMS_H
#define SDF2SDF_ALGORITHMS_H

#include "base.h"
#include "camera.h"
#include <memory>
#include <random>   // for generating normal distribution

class PnpBase {
private:
    std::default_random_engine generator;
    std::uniform_real_distribution<double> uniform_distribution;
    std::binomial_distribution<int> int_noise_distribution;
    void setRandomGeneratorSeed(int seed);
    /**
     * @brief Set up uniform distribution generator
     * @param lower
     * @param upper
     */
    void setUniformDistribution(dtype lower, dtype upper);
    void setBinomialDistribution(int upper, dtype p);
    dtype genNumFromDistribution(int method = 0);
    void genRect3dPair2d(const ImageBase *cam, std::vector<std::vector<Vec3>> &obj_points, std::vector<std::vector<Vec2>> &img_points, int num_sets = 1, int num_pairs = 4);
public:
    std::vector<Vec3> loadObjPointsFromFile(const std::string &file_path);
    std::vector<Vec2> loadImgPointsFromFile(const std::string &file_path);
    /**
     * @brief Select 4 image points from the image.
     * @param cam Camera Object
     * @return
     */
    std::vector<Vec2> selectImgPointsFromImage(const ImageBase *cam);
    bool saveImgPoints(const std::string &file_path, const std::vector<Vec2> &img_points);
    /**
     * @brief Compute reprojection error. MSRE is used.
     * @param cam
     * @param obj_points
     * @param img_points
     * @return
     */
    double reprojectError(const ImageBase *cam, const std::vector<Vec3> &obj_points, const std::vector<Vec2> &img_points);
    void noiseTest(ImageBase *cam, const std::string &path_obj_points, const std::string &path_img_points, const std::string &path_test_obj_points, const std::string &path_test_img_points);
    void noiseTeset(ImageBase *cam, dtype deviation = 5.);    
};

class P3P {
private:
    bool output_log = false;
    void P3P_LKneip(ImageBase *cam, const std::vector<Vec3> &obj_points, const std::vector<Vec2> &img_points);
    std::vector<Eigen::Matrix<dtype, 3, 4>> P3P_LKneipCandidates(const ImageBase *cam, const std::vector<Vec3> &obj_points, const std::vector<Vec2> &img_points);
    std::vector<dtype> solveQuartic(const dtype &a4, const dtype &a3, const dtype &a2, const dtype &a1, const dtype &a0);
public:
    void solve(ImageBase *cam, const std::vector<Vec3> &obj_points, const std::vector<Vec2> &img_points, int method = 0);
};

#endif //SDF2SDF_ALGORITHMS_H
