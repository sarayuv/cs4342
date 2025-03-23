
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

print('------- Problem 1 (40 points) --------')
# Total Points: 0.0




# Total Points: 0.0




# Total Points: 40.0

# ***********  Method: compute_gradient  (Class: Linear_SVM) **************** 
# Total Points: 20.0

total_points = test_function(1, "test_Linear_SVM_compute_gradient_test_2d", 2.0,total_points)
total_points = test_function(1, "test_Linear_SVM_compute_gradient_test_3d", 2.0,total_points)
total_points = test_function(1, "test_Linear_SVM_compute_gradient_test_b1", 2.0,total_points)
total_points = test_function(1, "test_Linear_SVM_compute_gradient_test_w1", 4.0,total_points)
total_points = test_function(1, "test_Linear_SVM_compute_gradient_test_l1", 2.0,total_points)
total_points = test_function(1, "test_Linear_SVM_compute_gradient_test_2d2", 4.0,total_points)
total_points = test_function(1, "test_Linear_SVM_compute_gradient_test_l2", 4.0,total_points)

# ***********  Method: train  (Class: Linear_SVM) **************** 
# Total Points: 20.0

total_points = test_function(1, "test_Linear_SVM_train_test_2d2s", 4.0,total_points)
total_points = test_function(1, "test_Linear_SVM_train_test_largeC", 4.0,total_points)
total_points = test_function(1, "test_Linear_SVM_train_test_smallC", 4.0,total_points)
total_points = test_function(1, "test_Linear_SVM_train_test_dataset", 8.0,total_points)






file_submit.write('problem2.py') 

print('------- Problem 2 (30 points) --------')
# Total Points: 30.0

# ***********  Method: compute_z  (Class: Logistic_Regression) **************** 
# Total Points: 1.5

total_points = test_function(2, "test_Logistic_Regression_compute_z_test_2d", 1.5,total_points)

# ***********  Method: compute_dz_db  (Class: Logistic_Regression) **************** 
# Total Points: 1.5

total_points = test_function(2, "test_Logistic_Regression_compute_dz_db_test_random", 1.5,total_points)


# ***********  Method: compute_dz_dw  (Class: Logistic_Regression) **************** 
# Total Points: 1.5

total_points = test_function(2, "test_Logistic_Regression_compute_dz_dw_test_random", 1.5,total_points)


# ***********  Method: compute_a  (Class: Logistic_Regression) **************** 
# Total Points: 3.0

total_points = test_function(2, "test_Logistic_Regression_compute_a_test_toy", 1.5,total_points)
total_points = test_function(2, "test_Logistic_Regression_compute_a_test_small_z", 1.5,total_points)

# ***********  Method: compute_da_dz  (Class: Logistic_Regression) **************** 
# Total Points: 1.5

total_points = test_function(2, "test_Logistic_Regression_compute_da_dz_test_toy", 0.75,total_points)
total_points = test_function(2, "test_Logistic_Regression_compute_da_dz_test_rand", 0.75,total_points)


# ***********  Method: compute_L  (Class: Logistic_Regression) **************** 
# Total Points: 3.0

total_points = test_function(2, "test_Logistic_Regression_compute_L_test_toy", 1.2,total_points)
total_points = test_function(2, "test_Logistic_Regression_compute_L_test_large_z", 0.9,total_points)
total_points = test_function(2, "test_Logistic_Regression_compute_L_test_negative_z", 0.9,total_points)

# ***********  Method: compute_dL_dz  (Class: Logistic_Regression) **************** 
# Total Points: 3.0

total_points = test_function(2, "test_Logistic_Regression_compute_dL_dz_test_toy", 0.6,total_points)
total_points = test_function(2, "test_Logistic_Regression_compute_dL_dz_test_large_z", 0.6,total_points)
total_points = test_function(2, "test_Logistic_Regression_compute_dL_dz_test_negative_z", 0.6,total_points)
total_points = test_function(2, "test_Logistic_Regression_compute_dL_dz_test_rand", 1.2,total_points)


# ***********  Method: compute_dL_db  (Class: Logistic_Regression) **************** 
# Total Points: 1.5

total_points = test_function(2, "test_Logistic_Regression_compute_dL_db_test_toy", 1.5,total_points)

# ***********  Method: compute_dL_dw  (Class: Logistic_Regression) **************** 
# Total Points: 1.5

total_points = test_function(2, "test_Logistic_Regression_compute_dL_dw_test_toy", 1.5,total_points)

# ***********  Method: backward  (Class: Logistic_Regression) **************** 
# Total Points: 3.0

total_points = test_function(2, "test_Logistic_Regression_backward_test_toy", 1.2,total_points)
total_points = test_function(2, "test_Logistic_Regression_backward_test_large_z", 1.8,total_points)

# ***********  Method: train  (Class: Logistic_Regression) **************** 
# Total Points: 9.0

total_points = test_function(2, "test_Logistic_Regression_train_test_2d4s", 2.7,total_points)
total_points = test_function(2, "test_Logistic_Regression_train_test_6samples", 2.7,total_points)
total_points = test_function(2, "test_Logistic_Regression_train_test_datafile", 3.6,total_points)






file_submit.write('problem3.py') 

print('------- Problem 3 (30 points) --------')
# Total Points: 0.0



# Total Points: 30.0

# ***********  Method: compute_z  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 1.5

total_points = test_function(3, "test_Logistic_Regression_Batch_compute_z_test_2d", 1.5,total_points)

# ***********  Method: compute_dz_db  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 1.5

total_points = test_function(3, "test_Logistic_Regression_Batch_compute_dz_db_test_1", 1.5,total_points)

# ***********  Method: compute_dz_dw  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 1.5

total_points = test_function(3, "test_Logistic_Regression_Batch_compute_dz_dw_test_1", 1.5,total_points)

# ***********  Method: compute_a  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 3.0

total_points = test_function(3, "test_Logistic_Regression_Batch_compute_a_test_toy", 3.0,total_points)

# ***********  Method: compute_da_dz  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 1.5

total_points = test_function(3, "test_Logistic_Regression_Batch_compute_da_dz_test_toy", 1.5,total_points)

# ***********  Method: compute_L  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 3.0

total_points = test_function(3, "test_Logistic_Regression_Batch_compute_L_test_toy", 3.0,total_points)

# ***********  Method: compute_dL_dz  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 3.0

total_points = test_function(3, "test_Logistic_Regression_Batch_compute_dL_dz_test_toy", 3.0,total_points)

# ***********  Method: compute_dL_db  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 1.5

total_points = test_function(3, "test_Logistic_Regression_Batch_compute_dL_db_test_toy", 1.5,total_points)

# ***********  Method: compute_dL_dw  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 1.5

total_points = test_function(3, "test_Logistic_Regression_Batch_compute_dL_dw_test_toy", 1.5,total_points)

# ***********  Method: backward  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 3.0

total_points = test_function(3, "test_Logistic_Regression_Batch_backward_test_toy", 3.0,total_points)

# ***********  Method: train  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 9.0

total_points = test_function(3, "test_Logistic_Regression_Batch_train_test_2d4s", 2.7,total_points)
total_points = test_function(3, "test_Logistic_Regression_Batch_train_test_6samples", 2.7,total_points)
total_points = test_function(3, "test_Logistic_Regression_Batch_train_test_datafile", 3.6,total_points)



file_submit.close()
print('****************************')
print(f'** Total Points: {round(total_points)} / 100  **')
print('****************************')
print('If you are good with this grade, you could submit your work in canvas. After running this grading script, a zip file named "submission.zip" was generated in the same folder of this homework assignment. This zip file is the only file that you need to submit in canvas. Thanks!')

