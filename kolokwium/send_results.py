import requests
import json

def send_results_to_server(results_data, server_url):
    try:
        headers = {'Content-Type': 'application/json'}
        # Przygotowanie danych w formacie JSON
        response = requests.post(server_url, data=json.dumps(results_data), headers=headers)
        
        # Sprawdzenie odpowiedzi serwera
        if response.status_code == 200:
            print("✅ Wyniki zostały pomyślnie wysłane na serwer.")
        else:
            print(f"❌ Wystąpił błąd podczas wysyłania wyników na serwer: {response.status_code}")
    except Exception as e:
        print(f"❌ Błąd podczas łączenia z serwerem: {e}")

