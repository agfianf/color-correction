import numpy as np

# in BGR format
reference_color_d50 = np.array(
    [
        [68, 82, 115],  # 1. Dark skin
        [128, 149, 195],  # 2. Light skin
        [157, 123, 93],  # 3. Blue sky
        [65, 108, 91],  # 4. Foliage
        [175, 129, 130],  # 5. Blue flower
        [171, 191, 99],  # 6. Bluish green
        [46, 123, 220],  # 7. Orange
        [168, 92, 72],  # 8. Purplish blue
        [97, 84, 194],  # 9. Moderate red
        [104, 59, 91],  # 10. Purple
        [62, 189, 161],  # 11. Yellow green
        [40, 161, 229],  # 12. Orange yellow
        [147, 63, 42],  # 13. Blue
        [72, 149, 72],  # 14. Green
        [57, 50, 175],  # 15. Red
        [22, 200, 238],  # 16. Yellow
        [150, 84, 188],  # 17. Magenta
        [166, 137, 0],  # 18. Cyan
        [240, 245, 245],  # 19. White 9.5
        [201, 202, 201],  # 20. Neutral 8
        [162, 162, 161],  # 21. Neutral 6.5
        [121, 121, 120],  # 22. Neutral 5
        [85, 85, 83],  # 23. Neutral 3.5
        [51, 50, 50],  # 24. Black 2
    ],
)
