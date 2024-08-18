def linear_interpolation(points, x):
    x0, y0 = points[0]
    x1, y1 = points[1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def polynomial_interpolation(points, x):
    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]

    a0 = y0
    a1 = (y1 - y0) / (x1 - x0)
    a2 = ((y2 - y1) / (x2 - x1) - a1) / (x2 - x0)

    return a0 + a1 * (x - x0) + a2 * (x - x0) * (x - x1)


def lagrange_interpolation(points, x):
    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]

    L0 = ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2))
    L1 = ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2))
    L2 = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1))

    return y0 * L0 + y1 * L1 + y2 * L2


def main():
    # Define points as pairs of (x, y)
    points = [(1, 2), (2, 4), (3, 6)]

    # x value for which we want to approximate y
    x = 2.5

    method = input("Choose interpolation method (linear, polynomial, lagrange): ").lower()

    if method == "linear":
        y = linear_interpolation(points[:2], x)  # Linear uses only the first two points
    elif method == "polynomial":
        y = polynomial_interpolation(points, x)  # Polynomial uses all three points
    elif method == "lagrange":
        y = lagrange_interpolation(points, x)  # Lagrange uses all three points
    else:
        print("Invalid method chosen.")
        return

    print(f"The interpolated value at x = {x} is y = {y}")


if __name__ == "__main__":
    main()