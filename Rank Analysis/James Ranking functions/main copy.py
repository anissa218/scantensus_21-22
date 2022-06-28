from flask import jsonify
from firebase.james import DummyRequest
from compare.compare import compare_answer, compare_config_write, compare_question


def main(request):
    print("Server has received a request of method: ", request.method)

    if request.method == "OPTIONS":
        print("This is a CORS preflight request being automatically sent by the user's browser "
              "before the real request comes. We have to respond to the user's browser that we are happy.")
        response = jsonify(success=True)
        response.headers.set("Access-Control-Allow-Origin", "*")
        response.headers.set("Access-Control-Allow-Credentials", True)
        response.headers.set("Access-Control-Allow-Methods", "GET,HEAD,OPTIONS,POST,PUT")
        response.headers.set("Access-Control-Allow-Headers", "Authorization, Content-Type")
        response.headers.set("Access-Control-Request-Headers", request.headers["Access-Control-Request-Headers"])
        return response

    else:
        json_data = request.get_json(force=True)
        action = json_data['action']
        print("This is the real request. Action is:", action)
        if action == "answer":
            response = compare_answer(request)
        elif action == "config_write":
            response = compare_config_write(request)
        elif action == "question":
            response = compare_question(request)
        else:
            return f"action should be 'answer', 'config_write', or 'question', not {action}"

        print(f"Response: {response}")
        response = jsonify(response)
        response.headers.set("Access-Control-Allow-Origin", "*")
        response.headers.set("Access-Control-Allow-Credentials", True)
        response.headers.set("Access-Control-Allow-Methods", "GET,HEAD,OPTIONS,POST,PUT")
        response.headers.set("Access-Control-Allow-Headers", "Authorization, Content-Type")
        return response
