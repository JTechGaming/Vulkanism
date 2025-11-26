#pragma once

#include <iostream>
#include <ostream>
#include <cmath>
#include <algorithm>
#include <numbers>
#include <vector>

// Name: Vulkanism -> Vulkan internal spectra math library

// todo: matrix<Y, Rows, Cols>, Quaternion<T>, Transform<T> (pos, rot, scale)
// todo: all the vulkan stuff

template <class T>
requires std::is_floating_point_v<T>
class Quaternion;

inline constexpr int byteAlignment = { 16 };

constexpr double degToRad(const double deg) { // idk if it can be a constexpr
    return deg*std::numbers::pi/180;
}

/*
 *  Vectors
 */
template <class T>
requires std::is_floating_point_v<T>
class Vector2 {
public:
    T x, y;

    // Constructors
    constexpr Vector2() : x(0), y(0) {}
    constexpr Vector2(T x_, T y_) : x(x_), y(y_) {}

    // Special initializers
    static constexpr Vector2 zero() { return Vector2(0,0); }
    static constexpr Vector2 one()  { return Vector2(1,1); }
    static constexpr Vector2 up()   { return Vector2(0,1); }
    static constexpr Vector2 right(){ return Vector2(1,0); }

    constexpr Vector2 add(const Vector2& other) const noexcept {
        return {x + other.x, y + other.y};
    }
    
    constexpr Vector2 subtract(const Vector2& other) const noexcept {
        return {x - other.x, y - other.y};
    }

    constexpr Vector2 multiply(const T scalar) const noexcept {
        return {x*scalar, y*scalar};
    }

    constexpr Vector2 divide(const T scalar) const noexcept {
        return {x/scalar, y/scalar};
    }

    constexpr Vector2 operator+(const Vector2& other) const noexcept {
        return add(other);
    }
    
    constexpr Vector2 operator-(const Vector2& other) const noexcept {
        return subtract(other);
    }

    constexpr Vector2& operator+=(const Vector2& other) noexcept {
        x += other.x;
        y += other.y;
        return *this;
    }

    constexpr Vector2& operator-=(const Vector2& other) noexcept {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    constexpr Vector2 operator*(const T scalar) const noexcept {
        return multiply(scalar);
    }

    constexpr Vector2 operator/(const T scalar) const noexcept {
        return divide(scalar);
    }

    constexpr Vector2& operator*=(const T scalar) noexcept {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    constexpr Vector2& operator/=(const T scalar) noexcept {
        x /= scalar;
        y /= scalar;
        return *this;
    }

    constexpr T dot(const Vector2& other) const noexcept {
        return x*other.x + y*other.y;
    } // dot
    
    constexpr T cross(const Vector2& other) const noexcept {
        return x*other.y - y*other.x;
    } // cross

    T distance(const Vector2& other) const noexcept {
        return std::sqrt((other.x-x)*(other.x-x)+(other.y-y)*(other.y-y));
    }

    constexpr T sqrMagnitude() const noexcept {
        return x*x + y*y;
    }
    
    T magnitude() const noexcept {
        return std::sqrt(sqrMagnitude());
    }
    
    Vector2 normalize() const noexcept {
        T mag = magnitude();
        if (mag == 0.0) {
            return {};
        }
        return {x/mag, y/mag};
    } // normalize / unit

    Vector2 reflect(const Vector2& surfaceUnitNormal) const noexcept {
        auto n = surfaceUnitNormal.normalize();
        return *this - n * (2 * dot(n));
    }

    Vector2 lerp(const Vector2& other, T t) const noexcept {
        return *this + t*(other-*this);
    }

    T angle(const Vector2& other) const noexcept {
        T denom = magnitude() * other.magnitude();
        if (denom == T(0)) return T(0);
        T c = dot(other) / denom;
        c = std::clamp(c, T(-1), T(1));
        return std::acos(c);
    } // angle

    T operator%(const Vector2& other) const noexcept {
        return angle(other);
    } // angle (overload)

    friend std::ostream& operator<<(std::ostream& os, const Vector2& v) {
        return os << "[" << v.x << " ; " << v.y << "]";
    }

    // Indexing
    T& operator[](size_t i) {
        switch (i) { case 0: return x; default: return y; }
    }

    const T& operator[](size_t i) const {
        switch (i) { case 0: return x; default: return y; }
    }

    // Iteration
    T* begin() { return &x; }
    T* end()   { return &x + 2; }

    const T* begin() const { return &x; }
    const T* end()   const { return &x + 2; }
};

template<std::floating_point T>
inline Vector2<T> operator*(T scalar, const Vector2<T>& v) noexcept {
    return v.multiply(scalar);
}

template <class T>
requires std::is_floating_point_v<T>
class Vector3 {
public:
    using value_type = T;

    T x, y, z;

    // Constructors
    constexpr Vector3() : x(0), y(0), z(0) {}
    constexpr Vector3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

    // Special initializers
    static constexpr Vector3 zero() { return Vector3(0,0, 0); }
    static constexpr Vector3 one()  { return Vector3(1,1, 1); }
    static constexpr Vector3 up()   { return Vector3(0,1, 0); }
    static constexpr Vector3 right(){ return Vector3(1,0, 0); }

    constexpr Vector3 add(const Vector3& other) const noexcept {
        return {x + other.x, y + other.y, z + other.z};
    }
    
    constexpr Vector3 subtract(const Vector3& other) const noexcept {
        return {x - other.x, y - other.y, z - other.z};
    }

    constexpr Vector3 multiply(const T scalar) const noexcept {
        return {x*scalar, y*scalar, z*scalar};
    }

    constexpr Vector3 divide(const T scalar) const noexcept {
        return {x/scalar, y/scalar, z/scalar};
    }

    constexpr Vector3 operator+(const Vector3& other) const noexcept {
        return add(other);
    }
    
    constexpr Vector3 operator-(const Vector3& other) const noexcept {
        return subtract(other);
    }

    constexpr Vector3& operator+=(const Vector3& other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    constexpr Vector3& operator-=(const Vector3& other) noexcept {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    constexpr Vector3 operator*(const T scalar) const noexcept {
        return multiply(scalar);
    }

    constexpr Vector3 operator/(const T scalar) const noexcept {
        return divide(scalar);
    }

    constexpr Vector3& operator*=(const T scalar) noexcept {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    constexpr Vector3& operator/=(const T scalar) noexcept {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    constexpr T dot(const Vector3& other) const noexcept {
        return x*other.x + y*other.y + z*other.z;
    } // dot
    
    constexpr Vector3 cross(const Vector3& other) const noexcept {
        return {
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        };
    } // cross x(a2b3 - b2a3) + y(a3b1 - a1b3) + z(a1b2 - a2b1)

    T distance(const Vector3& other) const noexcept {
        return std::sqrt((other.x-x)*(other.x-x)+(other.y-y)*(other.y-y)+(other.z-z)*(other.z-z));
    }

    constexpr T sqrMagnitude() const noexcept {
        return x*x + y*y + z*z;
    }
    
    T magnitude() const noexcept {
        return std::sqrt(sqrMagnitude());
    }
    
    Vector3 normalize() const noexcept {
        T mag = magnitude();
        if (mag == 0.0) {
            return {};
        }
        return {x/mag, y/mag, z/mag};
    } // normalize / unit

    Vector3 reflect(const Vector3& surfaceUnitNormal) const noexcept {
        auto n = surfaceUnitNormal.normalize();
        return *this - n * (2 * dot(n));
    }

    Vector3 lerp(const Vector3& other, T t) const noexcept {
        return *this + t*(other-*this);
    }

    T angle(const Vector3& other) const noexcept {
        T denom = magnitude() * other.magnitude();
        if (denom == T(0)) return T(0);
        T c = dot(other) / denom;
        c = std::clamp(c, T(-1), T(1));
        return std::acos(c);
    } // angle

    T operator%(const Vector3& other) const noexcept {
        return angle(other);
    } // angle (overload)

    friend std::ostream& operator<<(std::ostream& os, const Vector3& v) {
        return os << "[" << v.x << " ; " << v.y << " ; " << v.z << "]";
    }

    Vector2<T> toVector2() const {
        return {x, y};
    }

    /*
     *  Rotate this vector around the specified axis with the angle in degrees
     */
    Vector3<T> rotate(T angle, const Vector3<T>& axis);
    
    // Indexing
    T& operator[](size_t i) {
        switch (i) { case 0: return x; case 1: return y; default: return z; }
    }

    const T& operator[](size_t i) const {
        switch (i) { case 0: return x; case 1: return y; default: return z; }
    }

    // Iteration
    T* begin() { return &x; }
    T* end()   { return &x + 3; }

    const T* begin() const { return &x; }
    const T* end()   const { return &x + 3; }
};

template<std::floating_point T>
inline Vector3<T> operator*(T scalar, const Vector3<T>& v) noexcept {
    return v.multiply(scalar);
}

template <class T>
requires std::is_floating_point_v<T>
class Vector4 {
public:
    T x, y, z, w;

    // Constructors
    constexpr Vector4() : x(0), y(0), z(0), w(0) {}
    constexpr Vector4(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) {}

    // Special initializers
    static constexpr Vector4 zero() { return Vector4(0,0,0,0); }
    static constexpr Vector4 one()  { return Vector4(1,1,1,1); }
    static constexpr Vector4 up()   { return Vector4(0,1,0,1); }
    static constexpr Vector4 right(){ return Vector4(1,0,0,1); }

    constexpr Vector4 add(const Vector4& other) const noexcept {
        return {x + other.x, y + other.y, z + other.z, w + other.w};
    }
    
    constexpr Vector4 subtract(const Vector4& other) const noexcept {
        return {x - other.x, y - other.y, z - other.z, w - other.w};
    }

    constexpr Vector4 multiply(const T scalar) const noexcept {
        return {x*scalar, y*scalar, z*scalar, w*scalar};
    }

    constexpr Vector4 divide(const T scalar) const noexcept {
        return {x/scalar, y/scalar, z/scalar, w/scalar};
    }

    constexpr Vector4 operator+(const Vector4& other) const noexcept {
        return add(other);
    }
    
    constexpr Vector4 operator-(const Vector4& other) const noexcept {
        return subtract(other);
    }

    constexpr Vector4 operator-() const noexcept {
        return {-x, -y, -z, -w};
    }

    constexpr Vector4& operator+=(const Vector4& other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        w += other.w;
        return *this;
    }

    constexpr Vector4& operator-=(const Vector4& other) noexcept {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        w -= other.w;
        return *this;
    }

    constexpr Vector4 operator*(const T scalar) const noexcept {
        return multiply(scalar);
    }

    constexpr Vector4 operator/(const T scalar) const noexcept {
        return divide(scalar);
    }

    constexpr Vector4& operator*=(const T scalar) noexcept {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        w *= scalar;
        return *this;
    }

    constexpr Vector4& operator/=(const T scalar) noexcept {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        w /= scalar;
        return *this;
    }

    constexpr T dot(const Vector4& other) const noexcept {
        return x*other.x + y*other.y + z*other.z + w*other.w;
    } // dot

    T distance(const Vector4& other) const noexcept {
        return std::sqrt((other.x-x)*(other.x-x)+(other.y-y)*(other.y-y)+(other.z-z)*(other.z-z)+(other.w-w)*(other.w-w));
    }

    constexpr T sqrMagnitude() const noexcept {
        return x*x + y*y + z*z + w*w;
    }
    
    T magnitude() const noexcept {
        return std::sqrt(sqrMagnitude());
    }
    
    Vector4 normalize() const noexcept{
        T mag = magnitude();
        if (mag == 0.0) {
            return {};
        }
        return {x/mag, y/mag, z/mag, w/mag};
    } // normalize / unit

    Vector4 reflect(const Vector4& surfaceUnitNormal) const noexcept {
        auto n = surfaceUnitNormal.normalize();
        return *this - n * (2 * dot(n));
    }

    Vector4 lerp(const Vector4& other, T t) const noexcept {
        return *this + t*(other-*this);
    }

    T angle(const Vector4& other) const noexcept {
        T denom = magnitude() * other.magnitude();
        if (denom == T(0)) return T(0);
        T c = dot(other) / denom;
        c = std::clamp(c, T(-1), T(1));
        return std::acos(c);
    } // angle

    T operator%(const Vector4& other) const noexcept {
        return angle(other);
    } // angle (overload)

    friend std::ostream& operator<<(std::ostream& os, const Vector4& v) {
        return os << "[" << v.x << " ; " << v.y << " ; " << v.z << " ; " << v.w << "]";
    }

    Vector2<T> toVector2() const {
        return {x, y};
    }

    Vector3<T> toVector3() const {
        return {x, y, z};
    }

    // Indexing
    T& operator[](size_t i) {
        switch (i) { case 0: return x; case 1: return y; case 2: return z; default: return w; }
    }

    const T& operator[](size_t i) const {
        switch (i) { case 0: return x; case 1: return y; case 2: return z; default: return w; }
    }

    // Iteration
    T* begin() { return &x; }
    T* end()   { return &x + 4; }

    const T* begin() const { return &x; }
    const T* end()   const { return &x + 4; }
};

template<std::floating_point T>
inline Vector4<T> operator*(T scalar, const Vector4<T>& v) noexcept {
    return v.multiply(scalar);
}

/*
 *  Quaternions
 */
template <class T>
requires std::is_floating_point_v<T>
class Quaternion {
public:
    using value_type = T;
    T w;
    Vector3<T> v;

    constexpr Quaternion() : w(1), v(0,0,0) {}

    constexpr Quaternion(T w_, const Vector3<T>& v_)
        : w(w_), v(v_) {}

    constexpr Quaternion add(const Quaternion& other) const noexcept {
        return Quaternion(
            w+other.w, Vector3(
                v.x+other.v.x,
                v.y+other.v.y,
                v.z+other.v.z
            )
        );
    }

    constexpr Quaternion subtract(const Quaternion& other) const noexcept {
        return Quaternion(
            w-other.w, Vector3(
                v.x-other.v.x,
                v.y-other.v.y,
                v.z-other.v.z
            )
        );
    }
    
    constexpr Quaternion multiply(const T scalar) const noexcept {
        return Quaternion(
            w*scalar,Vector3(
                v.x*scalar,
                v.y*scalar,
                v.z*scalar
            )
        );
    }
    
    constexpr Quaternion product(const Quaternion& other) const noexcept {
        return Quaternion(
            w*other.w - v.x*other.v.x - v.y*other.v.y - v.z*other.v.z,
            Vector3<T>(
                w*other.v.x + v.x*other.w + v.y*other.v.z - v.z*other.v.y,
                w*other.v.y + v.y*other.w + v.z*other.v.x - v.x*other.v.z,
                w*other.v.z + v.z*other.w + v.x*other.v.y - v.y*other.v.x
            )
        );
    }

    constexpr T dot(const Quaternion& other) const noexcept {
        return w*other.w + v.dot(other.v);
    }

    constexpr Quaternion conjugate() const noexcept {
        return Quaternion(w, Vector3<T>(-v.x, -v.y, -v.z));
    }

    constexpr T squareNorm() const noexcept {
        return w*w+v.x*v.x+v.y*v.y+v.z*v.z;
    }
    
    constexpr T norm() const noexcept {
        return std::sqrt(squareNorm());
    }

    constexpr Quaternion normalize() const noexcept {
        return *this / norm();
    }

    constexpr Quaternion inverse() const noexcept {
        T n = squareNorm();
        return Quaternion(w/n, -v/n);
    }
    
    constexpr Quaternion difference(const Quaternion& other) const noexcept {
        return other * inverse();
    }

    constexpr T angularDifference(const Quaternion& other) const noexcept {
        return 2 * std::acos(difference(other).w);
    }

    constexpr Quaternion operator*(const Quaternion& other) const noexcept {
        return product(other);
    }

    constexpr Quaternion operator*(const T scalar) const noexcept {
        return multiply(scalar);
    }

    constexpr Quaternion operator+(const Quaternion& other) const noexcept {
        return add(other);
    }

    constexpr Quaternion operator-(const Quaternion& other) const noexcept {
        return subtract(other);
    }

    constexpr Quaternion slerp(const Quaternion& other, T time) {
        T angle = dot(other);

        // If quaternions are reversed, flip the second one
        Quaternion end = other;
        if (angle < 0) {
            angle = -angle;
            end = other * -1;
        }

        // If very close, do linear interpolation
        if (angle > 0.9995) {
            return (*this)*(1-time) + end*time;
        }

        T θ = std::acos(angle);
        T s = std::sin(θ);

        T w1 = std::sin((1-time)*θ) / s;
        T w2 = std::sin(time*θ) / s;

        return (*this)*w1 + end*w2;
    }
};

template <class T>
requires std::is_floating_point_v<T>
class Mat3x3 {
public:
    T a, b, c, d, e, f, g, h, i;
    // Constructors
    constexpr Mat3x3() : a(0), b(0), c(0), d(0), e(0), f(0), g(0), h(0), i(0) {}
    constexpr Mat3x3(T a, T b, T c, T d, T e, T f, T g, T h, T i) : a(a), b(b), c(c), d(d), e(e), f(f), g(g), h(h), i(i) {}

    friend std::ostream& operator<<(std::ostream& os, const Mat3x3& v) {
        return os
        << "[ " << v.a << "  " << v.b << "  " << v.c << "\n"
        << "  " << v.d << "  " << v.e << "  " << v.f << "\n"
        << "  " << v.g << "  " << v.h << "  " << v.i << " ]";
    }
};

template <class T, size_t Rows, size_t Cols>
requires std::is_floating_point_v<T>
class Mat {
public:
    std::array<std::array<T, Cols>, Rows> m{};

    constexpr T& operator()(size_t r, size_t c)
    {
        return m[r * Cols + c];
    }

    constexpr const T& operator()(size_t r, size_t c) const
    {
        return m[r * Cols + c];
    }

    constexpr Mat() = default;

    template<class... Args>
    constexpr Mat(Args... args)
    {
        static_assert(sizeof...(args) == Rows * Cols, "Initializer does not match matrix size");

        std::array<T, Rows * Cols> flat{ T(args)... };

        for (size_t r = 0; r < Rows; r++)
            for (size_t c = 0; c < Cols; c++)
                m[r][c] = flat[r * Cols + c];
    }
};

template <class T>
requires std::is_floating_point_v<T>
Vector3<T> Vector3<T>::rotate(T angle, const Vector3<T>& axis) {
    T half_angle = angle*std::numbers::pi/T(360); // 360 instead of 180 to bypass the additional division by 2
    T s = std::sin(half_angle);
    T c = std::cos(half_angle);
    Quaternion<T> q(c, axis.normalize() * s);
    Quaternion<T> p(T(0), *this);
    Quaternion<T> result = q * p * q.conjugate();
    return result.v;
}

inline int main(int argc, char* argv[]) {
    Vector2<float> vector2fA = Vector2<float>(2.5, 4.5);
    Vector2<float> vector2fB = Vector2<float>(4, 2);
    
    Vector2<float> vector2fC = vector2fA + vector2fB;
    std::cout << vector2fC.x << '\n';
    std::cout << vector2fC.y << '\n';
    std::cout << vector2fA.magnitude() << '\n';
    std::cout << vector2fA.reflect(vector2fB.normalize()) << '\n';

    Vector3<double> test = Vector3(1.0, 0.0, 0.0);
    std::cout << test.rotate(90.0, Vector3(0.0, 1.0, 0.0)) << '\n';

    Mat3x3<double> testMat3 = Mat3x3(1.0, 2.0, 6.0, 1.0, 0.0, 4.0, 1.5, 6.0, 1.0);
    std::cout << testMat3 << '\n';
    
    return 0;
}
