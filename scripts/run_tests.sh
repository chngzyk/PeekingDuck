#!/bin/bash
# author: Jin Jun
# date: 05 Jul 2021
#
# This shellscript controls the behavior of running/printing coverage 
# and the varying extent to run the test.

test_dir=$PWD/peekingduck
selectedTest="$1"
allowedExt=(all unit mlmodel module)


show_coverage(){
    # shows coverage if parameter is set to true and echos information.
    # showCoverage: type(bool) - shows coverage if set to true

    showCoverage=$1
    if [[ $showCoverage = true ]]; then
        echo "Coverage report printed." 
        coverage report
        echo "To hide the coverage report.. Set showCoverage to false"
    else
        echo "Coverage report not printed. To show the report set showCoverage to true"
    fi
}

run_test(){
    # runs pytest together with coverage, allows multiple configurations
    # showCoverage: type(bool) - whether to show coverage
    # testType: type(str) - type of test suite to run

    testType=$1
    showCoverage=$2

    if ! (coverage run --source="$test_dir" -m pytest -m "$testType"); then
        show_coverage $showCoverage
        echo "TEST FAILED."
        exit 1
    else
        show_coverage $showCoverage
        echo "TEST PASSED!"
    fi

    exit 0
}

echo "Running $selectedTest tests in $test_dir"

case $selectedTest in 
    "all")
        run_test "" true
        ;;
    "module")
        run_test "module" false
        ;;
    "mlmodel")
        run_test "mlmodel" false
        ;;
    "unit")
        run_test "not mlmodel and not module" false
        ;;
    *)
        echo "'$1' is an illegal argument, choose from: " "${allowedExt[@]}"
        exit 1
        ;;
esac
