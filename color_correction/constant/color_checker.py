import numpy as np

# in BGR format
reference_color_d50_bgr = np.array(
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

reference_color_d50_rgb = np.array(
    [
        [115, 82, 68],  # 1. Dark skin
        [195, 149, 128],  # 2. Light skin
        [93, 123, 157],  # 3. Blue sky
        [91, 108, 65],  # 4. Foliage
        [130, 129, 175],  # 5. Blue flower
        [99, 191, 171],  # 6. Bluish green
        [220, 123, 46],  # 7. Orange
        [72, 92, 168],  # 8. Purplish blue
        [194, 84, 97],  # 9. Moderate red
        [91, 59, 104],  # 10. Purple
        [161, 189, 62],  # 11. Yellow green
        [229, 161, 40],  # 12. Orange yellow
        [42, 63, 147],  # 13. Blue
        [72, 149, 72],  # 14. Green
        [175, 50, 57],  # 15. Red
        [238, 200, 22],  # 16. Yellow
        [188, 84, 150],  # 17. Magenta
        [0, 137, 166],  # 18. Cyan
        [245, 245, 240],  # 19. White 9.5
        [201, 202, 201],  # 20. Neutral 8
        [161, 162, 162],  # 21. Neutral 6.5
        [120, 121, 121],  # 22. Neutral 5
        [83, 85, 85],  # 23. Neutral 3.5
        [50, 50, 51],  # 24. Black 2
    ],
)
