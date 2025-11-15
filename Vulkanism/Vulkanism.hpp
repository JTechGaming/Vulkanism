#pragma once

#include <iostream>
#include <ostream>
#include <cmath>
#include <algorithm>

// Name: Vulkanism -> Vulkan internal spectra math library

// todo: matrix<Y, Rows, Cols>, Quaternion<T>, Transform<T> (pos, rot, scale)
// todo: all the vulkan stuff

inline constexpr int byteAlignment = { 16 };

template <class T>
requires std::is_floating_point_v<T>
class Vector2 {
public:
    using value_type = T;

    alignas(byteAlignment) T data[2];
    T& x = data[0];
    T& y = data[1];

    Vector2() : data{0,0}, x(data[0]), y(data[1]) {}
    Vector2(T x_, T y_) : data{x_,y_}, x(data[0]), y(data[1]) {}

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

    T& operator[](size_t index) { return data[index]; }
    const T& operator[](size_t index) const { return data[index]; }

    T* begin() { return data; }
    T* end()   { return data + 2; }

    const T* begin() const { return data; }
    const T* end()   const { return data + 2; }
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

    alignas(byteAlignment) T data[3];
    T& x = data[0];
    T& y = data[1];
    T& z = data[2];

    Vector3() : data{0,0,0}, x(data[0]), y(data[1]), z(data[2]) {}
    Vector3(T x_, T y_, T z_) : data{x_,y_,z_}, x(data[0]), y(data[1]), z(data[2]) {}

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

    T& operator[](size_t index) { return data[index]; }
    const T& operator[](size_t index) const { return data[index]; }

    T* begin() { return data; }
    T* end()   { return data + 3; }

    const T* begin() const { return data; }
    const T* end()   const { return data + 3; }
};

template<std::floating_point T>
inline Vector3<T> operator*(T scalar, const Vector3<T>& v) noexcept {
    return v.multiply(scalar);
}

template <class T>
requires std::is_floating_point_v<T>
class Vector4 {
public:
    using value_type = T;

    alignas(byteAlignment) T data[4];
    T& x = data[0];
    T& y = data[1];
    T& z = data[2];
    T& w = data[3];

    Vector4() : data{0,0,0,0}, x(data[0]), y(data[1]), z(data[2]), w(data[3]) {}
    Vector4(T x_, T y_, T z_, T w_) : data{x_,y_,z_,w_}, x(data[0]), y(data[1]), z(data[2]), w(data[3]) {}

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

    T& operator[](size_t index) { return data[index]; }
    const T& operator[](size_t index) const { return data[index]; }

    T* begin() { return data; }
    T* end()   { return data + 4; }

    const T* begin() const { return data; }
    const T* end()   const { return data + 4; }
};

template<std::floating_point T>
inline Vector4<T> operator*(T scalar, const Vector4<T>& v) noexcept {
    return v.multiply(scalar);
}

inline int main(int argc, char* argv[]) {
    Vector2<float> vector2fA = Vector2<float>(2.5, 4.5);
    Vector2<float> vector2fB = Vector2<float>(4, 2);
    
    Vector2<float> vector2fC = vector2fA + vector2fB;
    std::cout << vector2fC.x << '\n';
    std::cout << vector2fC.y << '\n';
    std::cout << vector2fA.magnitude() << '\n';
    std::cout << vector2fA.reflect(vector2fB.normalize()) << '\n';
    
    return 0;
}
