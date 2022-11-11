import json

from starlette.responses import Response
from fastapi import status


class APIResponse():
        def json_format(response=None, msg=''):
                if response != None: 
                        success = True
                        status_code = status.HTTP_200_OK
                        content = 'success' if msg == '' else msg 

                        return  Response(content=json.dumps(
                                                {
                                                'meta':{
                                                        'success': success,
                                                        'msg':content
                                                        },
                                                
                                                'response': response
                                                },
                                                ),
                                status_code=status_code,
                                media_type="application/json"
                                )

                else:
                        success = False
                        status_code = status.HTTP_501_NOT_IMPLEMENTED

                        content = 'Failed'
                return  Response(content=json.dumps(
                                                {
                                                'meta':{
                                                        'success': success,
                                                        'msg':content
                                                        }
                                                },
                                                ),
                                status_code=status_code,
                                media_type="application/json"
                        )

