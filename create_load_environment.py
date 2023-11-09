from os import path

LOAD_ENV_FILE = "load_environment.env"

def create_load_environment_file():
    with open(LOAD_ENV_FILE,"w") as env:
        env.write('OPENAI_API_KEY="INSERT KEY HERE"')
        env.write('HUGGINGFACE_API_KEY="INSERT KEY HERE"')
        env.write('CONFIG=configs')
if __name__ == "__main__":
    if path.isfile(LOAD_ENV_FILE):
        print("%s file already exists. Exiting" % LOAD_ENV_FILE)
    else:
        create_load_environment_file()
        print("%s file created. Add your API keys there" % LOAD_ENV_FILE)
