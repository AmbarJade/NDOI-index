# Steps to create an environment

```
python -m venv venv --prompt sel
```

# Steps to run environment

## First set policy in windows (just for this scope)

```
Set-ExecutionPolicy Unrestricted -Scope Process
```

## Then activate the environment

```
.\venv\Scripts\Activate
```


## Execute

```
python .\selector.py
```

## Install requeriments

```
python -m pip install --upgrade pip
pip install -r requirements.txt 
```
