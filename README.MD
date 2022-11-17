# Quick start

Requirements:

- Python 3.7

[Read the complete guide](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install-linux.html)

Clone this repository:

```bash
git https://github.com/cesarcacho1022/TESIS-II-DEV.git
```

Install libraries:

```bash
pip install -r requirements.txt
cd folder
```

## Start local server

```bash
flask run
```

## Make predictions

```bash
http://localhost:8080/test
```

## Deploy to EC2

```bash
tmux new -s flasksession
```

[Read the complete guide](https://levelup.gitconnected.com/how-to-deploy-a-flask-application-on-amazon-ec2-38837df3fa52)
