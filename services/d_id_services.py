import os
import requests
import json
import logging
import asyncio
import base64
from typing import Optional

logger = logging.getLogger(__name__)

class DIdService:
    """
    Service class to interact with the D-ID API for generating avatar videos from text.
    """
    def __init__(self, api_key: str, base_url: str = "https://api.d-id.com"):
        """
        Initializes the DIdService with the D-ID API key.

        Args:
            api_key (str): Your D-ID API key.
            base_url (str): The base URL for the D-ID API.
        """
        self.api_key = api_key
        self.base_url = base_url
        # D-ID uses Basic Authentication, where the API key is encoded in base64.
        self.headers = {
            "Authorization": f"Basic {base64.b64encode(self.api_key.encode()).decode()}",
            "accept": "application/json", # As per your provided headers
            "content-type": "application/json" # As per your provided headers
        }
        logger.info("DIdService initialized.")


    async def create_talk(self, script_text: str, 
                          presenter_image_url: str = "https://d-id-public-bucket.s3.us-west-2.amazonaws.com/alice.jpg", # Default to Alice
                          voice_id: str = "Sara", # Default to Sara voice
                          voice_provider: str = "microsoft",
                          voice_style: Optional[str] = None, # Optional voice style
                          subtitles: bool = False, # As per your provided payload
                          ssml: bool = False, # As per your provided payload
                          fluent: bool = False) -> dict: # As per your provided payload
        """
        Creates a D-ID 'talk' (video) from the given text script.

        Args:
            script_text (str): The text that the avatar should speak.
            presenter_image_url (str): The public URL of the image to use for the avatar.
                                       Defaults to D-ID's public Alice image.
            voice_id (str): The ID of the voice to use (e.g., 'Sara', 'en-US-JennyNeural').
            voice_provider (str): The provider of the voice (e.g., 'microsoft').
            voice_style (Optional[str]): The style of the voice (e.g., 'Cheerful').
            subtitles (bool): Whether to include subtitles.
            ssml (bool): Whether the input text is SSML.
            fluent (bool): Whether to use fluent mode for animation.

        Returns:
            dict: A dictionary containing the talk ID and status if successful,
                  or an 'error' key with details on failure.
        """
        url = f"{self.base_url}/talks"
        
        script_payload = {
            "type": "text",
            "input": script_text,
            "subtitles": str(subtitles).lower(), # D-ID expects "true" or "false" string
            "ssml": str(ssml).lower(),           # D-ID expects "true" or "false" string
            "provider": {
                "type": voice_provider,
                "voice_id": voice_id
            }
        }
        
        if voice_style: # Only add voice_config if a style is provided
            script_payload["provider"]["voice_config"] = {"style": voice_style}


        payload = {
            "source_url": presenter_image_url,
            "script": script_payload,
            "config": { 
                "fluent": str(fluent).lower() # D-ID expects "true" or "false" string
            }
        }
        
        # You could also add driver_url here if needed, e.g.:
        # payload["driver_url"] = "bank://lively/driver-05" 

        try:
            logger.info(f"Initiating D-ID talk creation. Presenter: {presenter_image_url}, Text: {script_text[:50]}...")
            logger.debug(f"D-ID Payload: {json.dumps(payload, indent=2)}") 

            response = await asyncio.to_thread(
                requests.post, url, headers=self.headers, data=json.dumps(payload)
            )
            response.raise_for_status() 
            result = response.json()
            logger.info(f"D-ID talk request successful. Talk ID: {result.get('id')}, Status: {result.get('status')}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error creating D-ID talk: {e}. Response: {e.response.text if e.response else 'N/A'}", exc_info=True)
            return {"error": str(e), "details": e.response.text if e.response else "No response text"}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error from D-ID response: {e}. Raw response: {response.text if 'response' in locals() else 'N/A'}", exc_info=True)
            return {"error": "Invalid JSON response", "details": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in DIdService.create_talk: {e}", exc_info=True)
            return {"error": "An unexpected error occurred", "details": str(e)}

    async def get_talk_status(self, talk_id: str) -> dict:
        """
        Retrieves the current status and result URL of a D-ID talk.

        Args:
            talk_id (str): The ID of the talk to check.

        Returns:
            dict: A dictionary containing the talk status, result URL (if available),
                  or an 'error' key on failure.
        """
        url = f"{self.base_url}/talks/{talk_id}"
        try:
            logger.debug(f"Getting D-ID talk status for ID: {talk_id}")
            response = await asyncio.to_thread(
                requests.get, url, headers=self.headers # Use self.headers which includes auth
            )
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Talk ID: {talk_id}, Current Status: {result.get('status')}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error getting D-ID talk status for {talk_id}: {e}. Response: {e.response.text if e.response else 'N/A'}", exc_info=True)
            return {"error": str(e), "details": e.response.text if e.response else "No response text"}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error from D-ID status response: {e}. Raw response: {response.text if 'response' in locals() else 'N/A'}", exc_info=True)
            return {"error": "Invalid JSON response", "details": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in DIdService.get_talk_status: {e}", exc_info=True)
            return {"error": "An unexpected error occurred", "details": str(e)}

# Import Optional for type hinting
from typing import Optional

# Example of how you might test this service (usually in main.py or a separate test file)
async def main():
    # This would typically come from your config.py
    # Ensure you have D_ID_API_KEY set in your .env file
    D_ID_API_KEY_EXAMPLE = os.getenv("D_ID_API_KEY") 
    if not D_ID_API_KEY_EXAMPLE:
        print("D_ID_API_KEY environment variable not set. Cannot run D-ID service example.")
        return

    d_id_service = DIdService(api_key=D_ID_API_KEY_EXAMPLE)

    # 1. Create a talk using the new parameters
    print("Creating a D-ID talk...")
    talk_creation_result = await d_id_service.create_talk(
        script_text="Hello, I am your AI interviewer. How are you today?",
        presenter_image_url="https://d-id-public-bucket.s3.us-west-2.amazonaws.com/alice.jpg", # Explicitly use Alice
        voice_id="Sara", # Explicitly use Sara voice
        voice_style="Cheerful", # Optional: Example of adding a style
        subtitles=False,
        ssml=False,
        fluent=True # Example: Enable fluent animation
    )

    if talk_creation_result.get("id"):
        talk_id = talk_creation_result["id"]
        print(f"Talk initiated with ID: {talk_id}")

        # 2. Poll for status until done
        status = ""
        video_url = None
        while status not in ["done", "error"]:
            print(f"Polling status for talk {talk_id}...")
            status_result = await d_id_service.get_talk_status(talk_id)
            status = status_result.get("status")
            video_url = status_result.get("result_url")
            print(f"Current status: {status}")
            if status == "done":
                print(f"Video ready! URL: {video_url}")
                # In a real application, you'd send this URL to your frontend
                break
            elif status == "error":
                print(f"Error generating video: {status_result.get('error', 'Unknown error')}")
                if status_result.get('details'):
                    print(f"Error details: {status_result['details']}")
                break
            await asyncio.sleep(2) # Wait for 2 seconds before polling again
    else:
        print(f"Failed to create talk: {talk_creation_result.get('error', 'Unknown error')}")
        if talk_creation_result.get('details'):
            print(f"Error details: {talk_creation_result['details']}")

if __name__ == "__main__":
    # To run this example, make sure you have a .env file with D_ID_API_KEY="your_key"
    # and run: python -m services.d_id_service
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())