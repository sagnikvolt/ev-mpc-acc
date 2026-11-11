# Deploying the demo (AWS, ap-south-1)

Everything is created by `template.yaml`; do not click resources together by hand. Cost is near zero at demo traffic (Lambda free tier, HTTP API pennies, logs kept 14 days) and a budget alert emails you at 80% of USD 1 per month.

## One-time prerequisites (your machine or AWS CloudShell)
- AWS CLI signed in to the right account, **AWS SAM CLI**, and **Docker** running (the Lambda is a container image because cvxpy/OSQP exceed the zip limit). CloudShell has the CLIs but not always enough disk for the Docker build; use your machine if it complains.

## Deploy
```
cd open_data_package/aws_demo
sam build --template-file template.yaml
sam deploy --guided --region ap-south-1 --stack-name ev-mpc-acc-demo \
  --parameter-overrides AlertEmail=YOUR_EMAIL MonthlyBudgetUSD=1
```
In the guided prompts accept the defaults, and answer **yes** to "Create managed ECR repositories". The `SimulateUrl` output is your API address.

## Connect the web page
1. Put that URL in `web/config.js` as `API_URL`.
2. Host the `web/` folder as a static site (GitHub Pages is fine).
3. Redeploy with `AllowedOrigin=https://YOUR-SITE` (instead of `*`) so only your page can call the API.

## Check
- `python3 test_local.py` (before deploying) must print PASS: it reruns paper windows through the handler and matches the stored results.
- After deploying: `curl -X POST "$SimulateUrl" -H "content-type: application/json" -d '{"speeds":[10,10,...]}'` or use the page.

## Take it down
`sam delete --stack-name ev-mpc-acc-demo` removes everything the template made.

## Safety limits already in the template
Throttle 1 request/s (burst 2); 28 s timeout; 30 to 180 s traces; no database, no stored uploads.
