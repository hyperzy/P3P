//
// Created by himalaya on 3/8/20 at 2:58 PM.
//

#ifndef SDF2SDF_BASE_H
#define SDF2SDF_BASE_H

/**
 *  Convention:
 *      Vec2 represents 2D point and 1st, 2nd entry correspond to x, y respectively
 *      Vec3 re[resents 3D point and 1st, 2nd, 3rd entry correspond to x, y, z respectively.
 *
 */
#include <opencv2/core.hpp>
#include <vector>
#include <Eigen/Eigen>//$PA
//#include <Eigen/V>

#define DEBUG_MODE
#undef DEBUG_MODE

typedef float dtype;
//typedef vtkFloatArray vtkDtypeArray;
//typedef cv::Point3f Point3;
typedef Eigen::Matrix<dtype, 3, 3, Eigen::RowMajor> Mat3;
typedef Eigen::Matrix<dtype, 4, 1> Vec4;
typedef Eigen::Matrix<dtype, 3, 1> Vec3;
typedef Eigen::Matrix<dtype, 2, 1> Vec2;
//typedef Eigen::
#define DTYPE CV_32F
#define DTYPEC1 CV_32FC1
constexpr dtype INF = 5e10;
#define FAR 1
typedef uint16_t d_bits;
typedef unsigned short IdxType;
typedef unsigned short DimUnit;

/*inline unsigned long Index(int i, int j, int k, int Y, int Z) {
    return i * Y * Z + j * Z + k;
}

inline unsigned long Index(int i, int j, int M) {
    return i * M + j;
}*/

#endif //SDF2SDF_BASE_H
