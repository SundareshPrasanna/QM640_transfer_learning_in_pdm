# C-MAPSS Data Description Report

This report provides a detailed statistical analysis of the raw C-MAPSS datasets (before normalization).

## Dataset: FD001

**Total Samples**: 20631

**Engines**: 100

**Constant Sensors (excluded)**: sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19

### 1. Descriptive Statistics

|           |     count |     mean |    std |   Variance |      min |      25% |      50% |      75% |      max |   Missing Count |   Missing % |
|:----------|----------:|---------:|-------:|-----------:|---------:|---------:|---------:|---------:|---------:|----------------:|------------:|
| sensor_2  | 20631.000 |  642.681 |  0.500 |      0.250 |  641.210 |  642.325 |  642.640 |  643.000 |  644.530 |           0.000 |       0.000 |
| sensor_3  | 20631.000 | 1590.523 |  6.131 |     37.591 | 1571.040 | 1586.260 | 1590.100 | 1594.380 | 1616.910 |           0.000 |       0.000 |
| sensor_4  | 20631.000 | 1408.934 |  9.001 |     81.011 | 1382.250 | 1402.360 | 1408.040 | 1414.555 | 1441.490 |           0.000 |       0.000 |
| sensor_6  | 20631.000 |   21.610 |  0.001 |      0.000 |   21.600 |   21.610 |   21.610 |   21.610 |   21.610 |           0.000 |       0.000 |
| sensor_7  | 20631.000 |  553.368 |  0.885 |      0.783 |  549.850 |  552.810 |  553.440 |  554.010 |  556.060 |           0.000 |       0.000 |
| sensor_8  | 20631.000 | 2388.097 |  0.071 |      0.005 | 2387.900 | 2388.050 | 2388.090 | 2388.140 | 2388.560 |           0.000 |       0.000 |
| sensor_9  | 20631.000 | 9065.243 | 22.083 |    487.654 | 9021.730 | 9053.100 | 9060.660 | 9069.420 | 9244.590 |           0.000 |       0.000 |
| sensor_11 | 20631.000 |   47.541 |  0.267 |      0.071 |   46.850 |   47.350 |   47.510 |   47.700 |   48.530 |           0.000 |       0.000 |
| sensor_12 | 20631.000 |  521.413 |  0.738 |      0.544 |  518.690 |  520.960 |  521.480 |  521.950 |  523.380 |           0.000 |       0.000 |
| sensor_13 | 20631.000 | 2388.096 |  0.072 |      0.005 | 2387.880 | 2388.040 | 2388.090 | 2388.140 | 2388.560 |           0.000 |       0.000 |
| sensor_14 | 20631.000 | 8143.753 | 19.076 |    363.900 | 8099.940 | 8133.245 | 8140.540 | 8148.310 | 8293.720 |           0.000 |       0.000 |
| sensor_15 | 20631.000 |    8.442 |  0.038 |      0.001 |    8.325 |    8.415 |    8.439 |    8.466 |    8.585 |           0.000 |       0.000 |
| sensor_17 | 20631.000 |  393.211 |  1.549 |      2.399 |  388.000 |  392.000 |  393.000 |  394.000 |  400.000 |           0.000 |       0.000 |
| sensor_20 | 20631.000 |   38.816 |  0.181 |      0.033 |   38.140 |   38.700 |   38.830 |   38.950 |   39.430 |           0.000 |       0.000 |
| sensor_21 | 20631.000 |   23.290 |  0.108 |      0.012 |   22.894 |   23.222 |   23.298 |   23.367 |   23.618 |           0.000 |       0.000 |

### 2. Visualizations

#### Sensor Distributions
![Distributions](figures/FD001_distributions.png)

#### Outlier Analysis (Boxplots)
![Boxplots](figures/FD001_boxplots.png)

#### Correlation Heatmap
![Correlation](figures/FD001_correlation.png)

### 3. Key Data Insights

- **High Correlation**: Strong linear relationship (>0.95) detected between: sensor_9 and sensor_14. This suggests redundancy; one sensor in each pair could likely be dropped without information loss.
- **Variability**: `sensor_9` shows the highest variance (487.65), indicating it fluctuates significantly across operating conditions. Conversely, `sensor_6` is the most stable.
- **Distribution Shape**: Sensors like `sensor_6` show high skewness (-6.92), indicating a non-normal distribution where extreme values are frequent.

---

## Dataset: FD002

**Total Samples**: 53759

**Engines**: 260

### 1. Descriptive Statistics

|           |     count |     mean |     std |   Variance |      min |      25% |      50% |      75% |      max |   Missing Count |   Missing % |
|:----------|----------:|---------:|--------:|-----------:|---------:|---------:|---------:|---------:|---------:|----------------:|------------:|
| sensor_1  | 53759.000 |  472.910 |  26.390 |    696.417 |  445.000 |  445.000 |  462.540 |  491.190 |  518.670 |           0.000 |       0.000 |
| sensor_2  | 53759.000 |  579.672 |  37.289 |   1390.499 |  535.530 |  549.570 |  555.980 |  607.340 |  644.520 |           0.000 |       0.000 |
| sensor_3  | 53759.000 | 1419.971 | 105.946 |  11224.627 | 1243.730 | 1352.760 | 1369.180 | 1499.370 | 1612.880 |           0.000 |       0.000 |
| sensor_4  | 53759.000 | 1205.442 | 119.123 |  14190.391 | 1023.770 | 1123.655 | 1138.890 | 1306.850 | 1439.230 |           0.000 |       0.000 |
| sensor_5  | 53759.000 |    8.032 |   3.614 |     13.060 |    3.910 |    3.910 |    7.050 |   10.520 |   14.620 |           0.000 |       0.000 |
| sensor_6  | 53759.000 |   11.601 |   5.432 |     29.504 |    5.710 |    5.720 |    9.030 |   15.490 |   21.610 |           0.000 |       0.000 |
| sensor_7  | 53759.000 |  282.607 | 146.005 |  21317.549 |  136.800 |  139.935 |  194.660 |  394.080 |  555.820 |           0.000 |       0.000 |
| sensor_8  | 53759.000 | 2228.879 | 145.210 |  21085.891 | 1914.770 | 2211.880 | 2223.070 | 2323.960 | 2388.390 |           0.000 |       0.000 |
| sensor_9  | 53759.000 | 8525.201 | 335.812 | 112769.708 | 7985.560 | 8321.660 | 8361.200 | 8778.030 | 9215.660 |           0.000 |       0.000 |
| sensor_10 | 53759.000 |    1.095 |   0.127 |      0.016 |    0.930 |    1.020 |    1.020 |    1.260 |    1.300 |           0.000 |       0.000 |
| sensor_11 | 53759.000 |   42.985 |   3.232 |     10.448 |   36.230 |   41.910 |   42.390 |   45.350 |   48.510 |           0.000 |       0.000 |
| sensor_12 | 53759.000 |  266.069 | 137.660 |  18950.140 |  129.120 |  131.520 |  183.200 |  371.260 |  523.370 |           0.000 |       0.000 |
| sensor_13 | 53759.000 | 2334.557 | 128.068 |  16401.482 | 2027.610 | 2387.900 | 2388.080 | 2388.170 | 2390.480 |           0.000 |       0.000 |
| sensor_14 | 53759.000 | 8066.598 |  84.838 |   7197.478 | 7848.360 | 8062.140 | 8082.540 | 8127.195 | 8268.500 |           0.000 |       0.000 |
| sensor_15 | 53759.000 |    9.330 |   0.749 |      0.562 |    8.336 |    8.678 |    9.311 |    9.387 |   11.067 |           0.000 |       0.000 |
| sensor_16 | 53759.000 |    0.023 |   0.005 |      0.000 |    0.020 |    0.020 |    0.020 |    0.030 |    0.030 |           0.000 |       0.000 |
| sensor_17 | 53759.000 |  348.310 |  27.755 |    770.313 |  303.000 |  331.000 |  335.000 |  369.000 |  399.000 |           0.000 |       0.000 |
| sensor_18 | 53759.000 | 2228.806 | 145.328 |  21120.222 | 1915.000 | 2212.000 | 2223.000 | 2324.000 | 2388.000 |           0.000 |       0.000 |
| sensor_19 | 53759.000 |   97.757 |   5.364 |     28.773 |   84.930 |  100.000 |  100.000 |  100.000 |  100.000 |           0.000 |       0.000 |
| sensor_20 | 53759.000 |   20.789 |   9.869 |     97.404 |   10.180 |   10.910 |   14.880 |   28.470 |   39.340 |           0.000 |       0.000 |
| sensor_21 | 53759.000 |   12.473 |   5.922 |     35.066 |    6.011 |    6.546 |    8.929 |   17.083 |   23.590 |           0.000 |       0.000 |

### 2. Visualizations

#### Sensor Distributions
![Distributions](figures/FD002_distributions.png)

#### Outlier Analysis (Boxplots)
![Boxplots](figures/FD002_boxplots.png)

#### Correlation Heatmap
![Correlation](figures/FD002_correlation.png)

### 3. Key Data Insights

- **High Correlation**: Strong linear relationship (>0.95) detected between: sensor_2 and sensor_3, sensor_2 and sensor_4, sensor_3 and sensor_4. This suggests redundancy; one sensor in each pair could likely be dropped without information loss.
- *Note: 46 other highly correlated pairs found.*
- **Variability**: `sensor_9` shows the highest variance (112769.71), indicating it fluctuates significantly across operating conditions. Conversely, `sensor_16` is the most stable.
- **Distribution Shape**: Sensors like `sensor_13` show high skewness (-1.97), indicating a non-normal distribution where extreme values are frequent.

---

## Dataset: FD003

**Total Samples**: 24720

**Engines**: 100

**Constant Sensors (excluded)**: sensor_1, sensor_5, sensor_16, sensor_18, sensor_19

### 1. Descriptive Statistics

|           |     count |     mean |    std |   Variance |      min |      25% |      50% |      75% |      max |   Missing Count |   Missing % |
|:----------|----------:|---------:|-------:|-----------:|---------:|---------:|---------:|---------:|---------:|----------------:|------------:|
| sensor_2  | 24720.000 |  642.458 |  0.523 |      0.274 |  640.840 |  642.080 |  642.400 |  642.790 |  645.110 |           0.000 |       0.000 |
| sensor_3  | 24720.000 | 1588.079 |  6.810 |     46.382 | 1564.300 | 1583.280 | 1587.520 | 1592.413 | 1615.390 |           0.000 |       0.000 |
| sensor_4  | 24720.000 | 1404.471 |  9.773 |     95.515 | 1377.060 | 1397.188 | 1402.910 | 1410.600 | 1441.160 |           0.000 |       0.000 |
| sensor_6  | 24720.000 |   21.596 |  0.018 |      0.000 |   21.450 |   21.580 |   21.600 |   21.610 |   21.610 |           0.000 |       0.000 |
| sensor_7  | 24720.000 |  555.144 |  3.437 |     11.815 |  549.610 |  553.110 |  554.050 |  556.040 |  570.490 |           0.000 |       0.000 |
| sensor_8  | 24720.000 | 2388.072 |  0.158 |      0.025 | 2386.900 | 2388.000 | 2388.070 | 2388.140 | 2388.600 |           0.000 |       0.000 |
| sensor_9  | 24720.000 | 9064.111 | 19.980 |    399.212 | 9017.980 | 9051.920 | 9060.010 | 9070.093 | 9234.350 |           0.000 |       0.000 |
| sensor_10 | 24720.000 |    1.301 |  0.003 |      0.000 |    1.290 |    1.300 |    1.300 |    1.300 |    1.320 |           0.000 |       0.000 |
| sensor_11 | 24720.000 |   47.416 |  0.300 |      0.090 |   46.690 |   47.190 |   47.360 |   47.600 |   48.440 |           0.000 |       0.000 |
| sensor_12 | 24720.000 |  523.051 |  3.255 |     10.597 |  517.770 |  521.150 |  521.980 |  523.840 |  537.400 |           0.000 |       0.000 |
| sensor_13 | 24720.000 | 2388.072 |  0.158 |      0.025 | 2386.930 | 2388.010 | 2388.070 | 2388.140 | 2388.610 |           0.000 |       0.000 |
| sensor_14 | 24720.000 | 8144.203 | 16.504 |    272.386 | 8099.680 | 8134.510 | 8141.200 | 8149.230 | 8290.550 |           0.000 |       0.000 |
| sensor_15 | 24720.000 |    8.396 |  0.061 |      0.004 |    8.156 |    8.361 |    8.398 |    8.437 |    8.570 |           0.000 |       0.000 |
| sensor_17 | 24720.000 |  392.567 |  1.761 |      3.103 |  388.000 |  391.000 |  392.000 |  394.000 |  399.000 |           0.000 |       0.000 |
| sensor_20 | 24720.000 |   38.989 |  0.249 |      0.062 |   38.170 |   38.830 |   38.990 |   39.140 |   39.850 |           0.000 |       0.000 |
| sensor_21 | 24720.000 |   23.393 |  0.149 |      0.022 |   22.873 |   23.296 |   23.392 |   23.483 |   23.951 |           0.000 |       0.000 |

### 2. Visualizations

#### Sensor Distributions
![Distributions](figures/FD003_distributions.png)

#### Outlier Analysis (Boxplots)
![Boxplots](figures/FD003_boxplots.png)

#### Correlation Heatmap
![Correlation](figures/FD003_correlation.png)

### 3. Key Data Insights

- **High Correlation**: Strong linear relationship (>0.95) detected between: sensor_7 and sensor_12, sensor_8 and sensor_13, sensor_9 and sensor_14. This suggests redundancy; one sensor in each pair could likely be dropped without information loss.
- **Variability**: `sensor_9` shows the highest variance (399.21), indicating it fluctuates significantly across operating conditions. Conversely, `sensor_10` is the most stable.
- **Distribution Shape**: Sensors like `sensor_6` show high skewness (-1.68), indicating a non-normal distribution where extreme values are frequent.

---

## Dataset: FD004

**Total Samples**: 61249

**Engines**: 249

### 1. Descriptive Statistics

|           |     count |     mean |     std |   Variance |      min |      25% |      50% |      75% |      max |   Missing Count |   Missing % |
|:----------|----------:|---------:|--------:|-----------:|---------:|---------:|---------:|---------:|---------:|----------------:|------------:|
| sensor_1  | 61249.000 |  472.882 |  26.437 |    698.906 |  445.000 |  445.000 |  462.540 |  491.190 |  518.670 |           0.000 |       0.000 |
| sensor_2  | 61249.000 |  579.420 |  37.343 |   1394.473 |  535.480 |  549.330 |  555.740 |  607.070 |  644.420 |           0.000 |       0.000 |
| sensor_3  | 61249.000 | 1417.897 | 106.168 |  11271.559 | 1242.670 | 1350.550 | 1367.680 | 1497.420 | 1613.000 |           0.000 |       0.000 |
| sensor_4  | 61249.000 | 1201.915 | 119.328 |  14239.074 | 1024.420 | 1119.490 | 1136.920 | 1302.620 | 1440.770 |           0.000 |       0.000 |
| sensor_5  | 61249.000 |    8.032 |   3.623 |     13.125 |    3.910 |    3.910 |    7.050 |   10.520 |   14.620 |           0.000 |       0.000 |
| sensor_6  | 61249.000 |   11.589 |   5.444 |     29.637 |    5.670 |    5.720 |    9.030 |   15.480 |   21.610 |           0.000 |       0.000 |
| sensor_7  | 61249.000 |  283.329 | 146.880 |  21573.796 |  136.170 |  142.920 |  194.960 |  394.280 |  570.810 |           0.000 |       0.000 |
| sensor_8  | 61249.000 | 2228.686 | 145.348 |  21126.112 | 1914.720 | 2211.950 | 2223.070 | 2323.930 | 2388.640 |           0.000 |       0.000 |
| sensor_9  | 61249.000 | 8524.673 | 336.928 | 113520.172 | 7984.510 | 8320.590 | 8362.760 | 8777.250 | 9196.810 |           0.000 |       0.000 |
| sensor_10 | 61249.000 |    1.096 |   0.128 |      0.016 |    0.930 |    1.020 |    1.030 |    1.260 |    1.320 |           0.000 |       0.000 |
| sensor_11 | 61249.000 |   42.875 |   3.243 |     10.520 |   36.040 |   41.760 |   42.330 |   45.220 |   48.360 |           0.000 |       0.000 |
| sensor_12 | 61249.000 |  266.736 | 138.479 |  19176.464 |  128.310 |  134.520 |  183.450 |  371.400 |  537.490 |           0.000 |       0.000 |
| sensor_13 | 61249.000 | 2334.428 | 128.198 |  16434.691 | 2027.570 | 2387.910 | 2388.060 | 2388.170 | 2390.490 |           0.000 |       0.000 |
| sensor_14 | 61249.000 | 8067.812 |  85.671 |   7339.442 | 7845.780 | 8062.630 | 8083.810 | 8128.350 | 8261.650 |           0.000 |       0.000 |
| sensor_15 | 61249.000 |    9.286 |   0.750 |      0.563 |    8.176 |    8.648 |    9.256 |    9.366 |   11.066 |           0.000 |       0.000 |
| sensor_16 | 61249.000 |    0.023 |   0.005 |      0.000 |    0.020 |    0.020 |    0.020 |    0.030 |    0.030 |           0.000 |       0.000 |
| sensor_17 | 61249.000 |  347.760 |  27.808 |    773.301 |  302.000 |  330.000 |  334.000 |  368.000 |  399.000 |           0.000 |       0.000 |
| sensor_18 | 61249.000 | 2228.613 | 145.472 |  21162.246 | 1915.000 | 2212.000 | 2223.000 | 2324.000 | 2388.000 |           0.000 |       0.000 |
| sensor_19 | 61249.000 |   97.751 |   5.369 |     28.831 |   84.930 |  100.000 |  100.000 |  100.000 |  100.000 |           0.000 |       0.000 |
| sensor_20 | 61249.000 |   20.864 |   9.936 |     98.732 |   10.160 |   10.940 |   14.930 |   28.560 |   39.890 |           0.000 |       0.000 |
| sensor_21 | 61249.000 |   12.519 |   5.963 |     35.554 |    6.084 |    6.566 |    8.960 |   17.136 |   23.885 |           0.000 |       0.000 |

### 2. Visualizations

#### Sensor Distributions
![Distributions](figures/FD004_distributions.png)

#### Outlier Analysis (Boxplots)
![Boxplots](figures/FD004_boxplots.png)

#### Correlation Heatmap
![Correlation](figures/FD004_correlation.png)

### 3. Key Data Insights

- **High Correlation**: Strong linear relationship (>0.95) detected between: sensor_2 and sensor_3, sensor_2 and sensor_4, sensor_3 and sensor_4. This suggests redundancy; one sensor in each pair could likely be dropped without information loss.
- *Note: 46 other highly correlated pairs found.*
- **Variability**: `sensor_9` shows the highest variance (113520.17), indicating it fluctuates significantly across operating conditions. Conversely, `sensor_16` is the most stable.
- **Distribution Shape**: Sensors like `sensor_13` show high skewness (-1.97), indicating a non-normal distribution where extreme values are frequent.

---

