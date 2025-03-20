
import pytest 
import zipfile 
from pytest import ExitCode

#------------------------------------------
# AutoGrading of HW assignment
# How to run?  In the terminal, type:
#  (Windows OS) python grading.py
#  (Mac OS & Linux) python3 grading.py
#------------------------------------------

def test_function(pid,name, point,total_points):
    result = pytest.main(["--no-header","--tb=no",f"test_{pid}.py::{name}"])
    if result == ExitCode.OK:
        total_points += point
        print(f'*** Pass ({point} pt) --- {name}')
    else:
        print(f'*** Fail (0 / {point} pt) --- {name}')
    return total_points

total_points = 0
file_submit = zipfile.ZipFile('submission.zip','w')


file_submit.write('problem1.py') 

print('------- Problem 1 (20 points) --------')
# Total Points: 20.0

# ***********  Method: find_neighbor  (Class: Nearest_Neighbor) **************** 
# Total Points: 10.0

total_points = test_function(1, "test_Nearest_Neighbor_find_neighbor_test_2d", 5.0,total_points)
total_points = test_function(1, "test_Nearest_Neighbor_find_neighbor_test_3d", 5.0,total_points)

# ***********  Method: predict  (Class: Nearest_Neighbor) **************** 
# Total Points: 10.0

total_points = test_function(1, "test_Nearest_Neighbor_predict_test_2d", 5.0,total_points)
total_points = test_function(1, "test_Nearest_Neighbor_predict_test_3d", 5.0,total_points)






file_submit.write('problem2.py') 

print('------- Problem 2 (20 points) --------')
# Total Points: 0.0



# Total Points: 10.0
# ***********  Method: train  (Class: Simple_Regression_Abs) **************** 
# Total Points: 10.0

total_points = test_function(2, "test_Simple_Regression_Abs_train_test_3instances", 3.0,total_points)
total_points = test_function(2, "test_Simple_Regression_Abs_train_test_5instances", 3.0,total_points)
total_points = test_function(2, "test_Simple_Regression_Abs_train_test_7instances", 4.0,total_points)

# Total Points: 10.0
# ***********  Method: train  (Class: Simple_Regression_SE) **************** 
# Total Points: 10.0

total_points = test_function(2, "test_Simple_Regression_SE_train_test_3instances", 3.0,total_points)
total_points = test_function(2, "test_Simple_Regression_SE_train_test_5instances", 3.0,total_points)
total_points = test_function(2, "test_Simple_Regression_SE_train_test_7instances", 4.0,total_points)






file_submit.write('problem3.py') 

print('------- Problem 3 (20 points) --------')
# Total Points: 0.0



# Total Points: 10.0
# ***********  Method: train  (Class: Linear_Regression_SE) **************** 
# Total Points: 10.0

total_points = test_function(3, "test_Linear_Regression_SE_train_test_3instances", 5.0,total_points)
total_points = test_function(3, "test_Linear_Regression_SE_train_test_random", 5.0,total_points)

# Total Points: 10.0

# ***********  Method: train  (Class: Linear_Regression_Ridge) **************** 
# Total Points: 10.0

total_points = test_function(3, "test_Linear_Regression_Ridge_train_test_3instances", 5.0,total_points)
total_points = test_function(3, "test_Linear_Regression_Ridge_train_test_random", 5.0,total_points)






file_submit.write('problem4.py') 

print('------- Problem 4 (40 points) --------')
# Total Points: 40.0

# ***********  Method: compute_gradient  (Class: Lasso_Regression) **************** 
# Total Points: 20.0

total_points = test_function(4, "test_Lasso_Regression_compute_gradient_test_2d", 4.0,total_points)
total_points = test_function(4, "test_Lasso_Regression_compute_gradient_test_3d", 4.0,total_points)
total_points = test_function(4, "test_Lasso_Regression_compute_gradient_test_alpha", 4.0,total_points)
total_points = test_function(4, "test_Lasso_Regression_compute_gradient_test_weight", 4.0,total_points)
total_points = test_function(4, "test_Lasso_Regression_compute_gradient_test_label", 4.0,total_points)

# ***********  Method: update_w  (Class: Lasso_Regression) **************** 
# Total Points: 8.0

total_points = test_function(4, "test_Lasso_Regression_update_w_test_2d", 2.4,total_points)
total_points = test_function(4, "test_Lasso_Regression_update_w_test_3d", 2.4,total_points)
total_points = test_function(4, "test_Lasso_Regression_update_w_test_lr", 3.2,total_points)

# ***********  Method: train  (Class: Lasso_Regression) **************** 
# Total Points: 12.0

total_points = test_function(4, "test_Lasso_Regression_train_test_alpha0", 3.6,total_points)
total_points = test_function(4, "test_Lasso_Regression_train_test_large_alpha", 4.8,total_points)
total_points = test_function(4, "test_Lasso_Regression_train_test_medium_alpha", 3.6,total_points)



file_submit.close()
print('****************************')
print(f'** Total Points: {round(total_points)} / 100  **')
print('****************************')
print('If you are good with this grade, you could submit your work in canvas. After running this grading script, a zip file named "submission.zip" was generated in the same folder of this homework assignment. This zip file is the only file that you need to submit in canvas. Thanks!')

