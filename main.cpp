#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Eigen>

#include <cmath>
#include <random>
#include <utility>

using SysClock = std::chrono::system_clock;
using Timestamp = SysClock::time_point;
template <typename T> using Duration = std::chrono::duration<T>;
template <typename T> using Vector = std::vector<T>;

struct GraphicLine
{
    cv::Point p1, p2;
    int label;
};

struct GeometryLine
{
    Eigen::Vector2d p1, p2;
    int label = 0;

    GraphicLine graphic(const cv::Size & imageSize) const
    {
        return {
            cv::Point(
                (int) p1(0) + imageSize.width / 2,
                (int) -p1(1) + imageSize.height / 2
            ),
            cv::Point(
                (int) p2(0) + imageSize.width / 2,
                (int) -p2(1) + imageSize.height / 2
            ),
            label
        };
    }

    GeometryLine transform(const Eigen::Matrix2d & mat) const
    {
        return GeometryLine(mat * p1, mat * p2, label);
    }

    GeometryLine() = default;

    explicit GeometryLine(Eigen::Vector2d p1, Eigen::Vector2d p2, int label)
        : p1(std::move(p1)), p2(std::move(p2)), label(label)
    { /* empty */ }

    explicit GeometryLine(
        double x1, double y1, double x2, double y2,
        int label = 0
    ) {
        p1 << x1, y1;
        p2 << x2, y2;
        this->label = label;
    }
};

class Geometry
{
  private:

    Vector<GeometryLine> lines;

  public:

    void rotate(double angle)
    {
        Eigen::Matrix2d rotateMatrix;
        rotateMatrix << std::cos(angle), std::sin(angle),
                        -std::sin(angle), std::cos(angle);
        for (GeometryLine & line : lines) {
            line = line.transform(rotateMatrix);
        }
    }

    void scale(double scl)
    {
        for (GeometryLine & line : lines) {
            line.p1 *= scl;
            line.p2 *= scl;
        }
    }

    void addLine(const GeometryLine & line)
    {
        this->lines.emplace_back(line);
    }

    Vector<GraphicLine> getGraphic(const cv::Size & imageSize) const
    {
        Vector<GraphicLine> graphicLines;
        for (const GeometryLine & line : lines) {
            graphicLines.emplace_back(line.graphic(imageSize));
        }
        return graphicLines;
    }

    Geometry() = default;

    explicit Geometry(Vector<GeometryLine> lines)
        : lines(std::move(lines))
    { /* empty */ }
};

enum LightBlobLabel
{
    LIGHT_BLOB_MASK             = 0,
    LIGHT_BLOB_ARMOR_HORIZONTAL = 1,
    LIGHT_BLOB_ARMOR_VERTICAL   = 2,
    LIGHT_BLOB_ARM_STICK        = 3,
    LIGHT_BLOB_OUTSIDE          = 4,
};

class PowerRuneArm
{
  public:

    enum State
    {
        DEACTIVATED = 0,
        ACTIVATING = 1,
        ACTIVATED = 2,
    };

  private:

    const Geometry geometry = Geometry({
        GeometryLine(0, 20, 0, 66,      LIGHT_BLOB_ARM_STICK),
        GeometryLine(-13, 85, 13, 85,   LIGHT_BLOB_ARMOR_HORIZONTAL),
        GeometryLine(-14, 68, 14, 68,   LIGHT_BLOB_ARMOR_HORIZONTAL),
        GeometryLine(-13, 85, -14, 68,  LIGHT_BLOB_ARMOR_VERTICAL),
        GeometryLine(13, 85, 14, 68,    LIGHT_BLOB_ARMOR_VERTICAL),
        GeometryLine(-6, 18, 6, 18,     LIGHT_BLOB_OUTSIDE),  // `_`
        GeometryLine(6, 18, 16, 31,     LIGHT_BLOB_OUTSIDE),  // `/`
        GeometryLine(-6, 18, -16, 31,   LIGHT_BLOB_OUTSIDE),  // `\`
        GeometryLine(16, 31, 14, 68,    LIGHT_BLOB_OUTSIDE),  // `|`
        GeometryLine(-16, 31, -14, 68,  LIGHT_BLOB_OUTSIDE),  // `|`
    });

    State state = DEACTIVATED;
    Timestamp createdTime = SysClock::now();

    static double maskYOffset(double t)
    {
        t *= 2;
        return 6 * (t - floor(t));
    }

  public:

    int id = 0;

    State activateState() const
    {
        return this->state;
    }

    void light()
    {
        state = ACTIVATING;
    }

    void activate()
    {
        state = ACTIVATED;
    }

    void deactivate()
    {
        state = DEACTIVATED;
    }

    Geometry getStickMask(double angle = 0, double scale = 1) const
    {
        double t = Duration<double>(SysClock::now() - createdTime).count();
        double offset = maskYOffset(t);
        GeometryLine lineLeft  = GeometryLine(0, 6 + offset, -6, 0 + offset, LIGHT_BLOB_MASK);
        GeometryLine lineRight = GeometryLine(0, 6 + offset,  6, 0 + offset, LIGHT_BLOB_MASK);

        Geometry mask;

        for (int i = 0; i < 10; i++) {
            lineLeft.p1(1)  += 6;
            lineLeft.p2(1)  += 6;
            lineRight.p1(1) += 6;
            lineRight.p2(1) += 6;
            mask.addLine(lineLeft);
            mask.addLine(lineRight);
        }

        mask.rotate(this->id * CV_PI * (2.0 / 5.0) + angle);
        mask.scale(scale);

        return mask;
    }

    Geometry getGeometry(double angle = 0, double scale = 1) const
    {
        Geometry geo = this->geometry;
        geo.rotate(this->id * CV_PI * (2.0 / 5.0) + angle);
        geo.scale(scale);
        return geo;
    }

    PowerRuneArm() = default;

    PowerRuneArm(const PowerRuneArm & arm)
    {
        this->id = arm.id;
        this->state = arm.state;
    }

};

class ClockSin
{
  private:

    Timestamp lastTimeStamp = SysClock::now();
    // t: sec
    double t = 0;

  public:

    double A, w, b;
    ClockSin(double a, double w, double b) : A(a), w(w), b(b) {}

    double operator() ()
    {
        Timestamp current = SysClock::now();
        this->t += Duration<double>(current - lastTimeStamp).count();
        double value = A * std::sin(w * t) + b;
        lastTimeStamp = current;
        return value;
    }

    double integral()
    {
        Timestamp current = SysClock::now();
        double deltaT = Duration<double>(current - lastTimeStamp).count();
        t += deltaT;
        double value = (A * std::sin(w * t) + b) * deltaT;
        lastTimeStamp = current;
        return value;
    }
};

enum Color
{
    RED,
    BLUE,
};

class PowerRune
{
  private:

    PowerRuneArm arms[5];
    std::mt19937 mt = std::mt19937(std::random_device()());
    int activeOrder[5] = { 0, 1, 2, 3, 4 };
    int activating = 0;
    double angle = 0;
    double scale = 3;
    ClockSin clockSin;
    Color color;
    double bloomFactor = 1.4;

    void next()
    {
        this->angle += clockSin.integral();
    }

    cv::Mat bloom(const cv::Mat & src) const
    {
        cv::Mat gaussian, image;
        cv::GaussianBlur(src, image, cv::Size(3, 3), 1);
        int gaussianKernelSize = (int) (9.0 * scale);
        gaussianKernelSize += gaussianKernelSize % 2 == 0 ? 1 : 0;
        cv::GaussianBlur(image, gaussian, cv::Size(gaussianKernelSize, gaussianKernelSize), 8);
        return image + gaussian * bloomFactor;
    }

  public:

    cv::Mat renderImage()
    {
        this->next();
        auto imageSize = cv::Size((int) (scale * 200), (int) (scale * 200));
        cv::Mat image = cv::Mat::zeros(imageSize, CV_8UC3);
        cv::Scalar rgbColor = this->color == RED ? cv::Scalar(50, 70, 255) : cv::Scalar(255, 70, 50);

        for (const auto & arm : arms) {
            auto geo = arm.getGeometry(this->angle, this->scale);
            for (const GraphicLine & line : geo.getGraphic(image.size())) {
                switch (line.label) {
                    case LIGHT_BLOB_ARMOR_HORIZONTAL: {
                        if (arm.activateState() != PowerRuneArm::DEACTIVATED) {
                            cv::line(image, line.p1, line.p2, rgbColor, scale * 2, cv::LINE_AA);
                        }
                    } break;

                    case LIGHT_BLOB_ARMOR_VERTICAL: {
                        if (arm.activateState() != PowerRuneArm::DEACTIVATED) {
                            cv::line(image, line.p1, line.p2, rgbColor, scale * 1.5, cv::LINE_AA);
                        }
                    } break;

                    case LIGHT_BLOB_ARM_STICK: {
                        if (arm.activateState() == PowerRuneArm::ACTIVATED) {
                            cv::line(image, line.p1, line.p2, rgbColor, scale * 4, cv::LINE_AA);
                        } else if (arm.activateState() == PowerRuneArm::ACTIVATING) {
                            cv::line(image, line.p1, line.p2, rgbColor, scale * 4, cv::LINE_AA);
                            for (GraphicLine & mask : arm.getStickMask(this->angle, this->scale).getGraphic(imageSize)) {
                                const cv::Scalar BLACK(0, 0, 0);
                                cv::line(image, mask.p1, mask.p2, BLACK, scale * 2.4, cv::LINE_AA);
                            }
                        }
                    } break;

                    case LIGHT_BLOB_OUTSIDE: {
                        if (arm.activateState() == PowerRuneArm::ACTIVATED) {
                            cv::line(image, line.p1, line.p2, rgbColor, scale * 1, cv::LINE_AA);
                        }
                    } break;
                }
            }
        }
        return bloom(image);
    }

    void switchColor()
    {
        if (this->color == RED) {
            this->color = BLUE;
        } else {
            this->color = RED;
        }
    }

    void setBloom(double factor)
    {
        this->bloomFactor = factor;
    }

    void setScale(double value)
    {
        this->scale = value;
    }

    void setSin_A(double value)
    {
        this->clockSin.A = value;
    }

    void setSin_w(double value)
    {
        this->clockSin.w = value;
    }

    void setSin_b(double value)
    {
        this->clockSin.b = value;
    }

    void activateNext()
    {
        if (activating == 6) {
            std::shuffle(activeOrder, activeOrder + 5, mt);
            for (auto & arm : arms) {
                arm.deactivate();
            }
            activating = 0;
            arms[activeOrder[0]].light();
        } else {
            arms[activeOrder[activating++]].activate();
            if (activating == 5) {
                activating += 1;
            } else {
                arms[activeOrder[activating]].light();
            }
        }
    }

    explicit PowerRune(double A, double w, double b, Color color)
        noexcept : clockSin(A, w, b), color(color)
    {
        for (int i = 0; i < 5; i++) {
            arms[i].id = i;
            arms[i].activate();
        }
        std::shuffle(activeOrder, activeOrder + 5, mt);
        activating = 6;
    }

};

PowerRune powerRune(0.785, 1.884, 1.305, RED);

void onMouseClick(int event, int, int, int, void*)
{
    if (event == cv::EVENT_LBUTTONDOWN) {
        powerRune.activateNext();
    }
}

void onColorChange(int, void*)
{
    powerRune.switchColor();
}

void onBloomChange(int value, void*)
{
    powerRune.setBloom(0.5 + value / 25.0);
}

void onSizeChange(int value, void*)
{
    powerRune.setScale(value * 0.1 + 1);
}


#define WIN_NAME "PowerRune :: A sin(wt + b)"
#define SIN_TRACKBAR(var_, expr_)                                                       \
    cv::createTrackbar(#var_" "#expr_, WIN_NAME, nullptr, 100, [](int (var_), void*) {  \
        powerRune.setSin_##var_ (var_ expr_);                                           \
    })                                                                                  \

void initializeWindow()
{
    cv::namedWindow(WIN_NAME);
    cv::setMouseCallback(WIN_NAME, onMouseClick);

    SIN_TRACKBAR(A, * 0.05);
    SIN_TRACKBAR(w, * 0.05);
    SIN_TRACKBAR(b, * 0.1 - 5);

    cv::createTrackbar("color", WIN_NAME, nullptr, 1, onColorChange);
    cv::createTrackbar("brightness", WIN_NAME, nullptr, 100, onBloomChange);
    cv::createTrackbar("size", WIN_NAME, nullptr, 300, onSizeChange);
}

[[noreturn]]
int main()
{
    initializeWindow();
    while (true) {
        cv::imshow(WIN_NAME, powerRune.renderImage());
        cv::waitKey(1);
    }
}
