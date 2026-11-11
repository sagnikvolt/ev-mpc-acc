# Build context is the PACKAGE ROOT (see template.yaml DockerContext). Lambda container image, because cvxpy/OSQP exceed the zip size limit.
FROM public.ecr.aws/lambda/python:3.12
COPY aws_demo/app/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt
COPY aws_demo/app/handler.py ${LAMBDA_TASK_ROOT}/handler.py
COPY code/python/mpc_acc.py code/python/baselines.py ${LAMBDA_TASK_ROOT}/code/python/
COPY derived_data/bmw_i3_params_final.csv ${LAMBDA_TASK_ROOT}/derived_data/bmw_i3_params_final.csv
CMD ["handler.lambda_handler"]
