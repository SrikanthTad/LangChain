import os
import requests

L_API_KEY = ""

def scrape_linkedin_profile(linkedin_profile_url: str):

    """scrape information from linkedIn profile"""
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin" #api
    header_dic = {"Authorization": f'Bearer {os.environ.get("L_API_KEY")}'} #key of authorization

    response = requests.get(api_endpoint, params = {"url": linkedin_profile_url}, headers = header_dic)
    #To save on token usage, we don't pull empty values from the json

    data = response.json()
    data = {k: v
            for k , v in data.items()
            if v not in ([],"","", None)
                and k not in ["people_also_viewed", "certifications"] #too many values so remove
            }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url") #remove also

    return data