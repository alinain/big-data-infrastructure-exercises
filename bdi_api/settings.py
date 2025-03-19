from os.path import dirname, join

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import bdi_api

PROJECT_DIR = dirname(dirname(bdi_api.__file__))


class DBCredentials(BaseSettings):
    """Use env variables prefixed with BDI_DB_"""
    #PostgreSQL connection details
    host: str = "my-database.cxbhrx6lorso.us-east-1.rds.amazonaws.com" 
    port: int = 5432  
    username: str = "postgres"  
    password: str = "DieToreDerWelt"  
    database: str = "my-database"  

   # host: str
   # port: int = 5432
   # username: str
   # password: str
   # database: str 

    model_config = SettingsConfigDict(env_prefix="bdi_db_")


class Settings(BaseSettings):
    source_url: str = Field(
        default="https://samples.adsbexchange.com/readsb-hist",
        description="Base URL to the website used to download the data.",
    )
    local_dir: str = Field(
        default=join(PROJECT_DIR, "data"),
        description="For any other value set env variable 'BDI_LOCAL_DIR'",
    )
    s3_bucket: str = Field(
        default="bdi-test",
        description="Call the api like `BDI_S3_BUCKET=yourbucket poetry run uvicorn...`",
    )


    telemetry: bool = False
    telemetry_dsn: str = "http://project2_secret_token@uptrace:14317/2"

    model_config = SettingsConfigDict(env_prefix="bdi_")

    @property
    def raw_dir(self) -> str:
        """Store inside all the raw jsons"""
        return join(self.local_dir, "raw")

    @property
    def prepared_dir(self) -> str:
        return join(self.local_dir, "prepared")
