from rag import VoiceOfCustomerRAG

def main():
    print("\nVOICE OF CUSTOMER INTELLIGENCE — OLIST")
    print("Digite uma pergunta sobre as avaliações. Para sair, digite 'sair'.")
    rag = VoiceOfCustomerRAG()

    while True:
        question = input("\nPergunta: ").strip()
        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando.")
            break
        if not question:
            continue

        try:
            answer, evidence = rag.answer(question)
            print("\nRESPOSTA")
            print(answer)
            print("\nEVIDÊNCIAS RECUPERADAS")
            if not evidence:
                print("Nenhuma evidência atingiu o critério mínimo de similaridade.")
            for item in evidence:
                print(
                    f"- Review ID: {item['review_id']} | Nota: {item['score']} | "
                    f"Data: {item['date']} | Distância: {item['distance']:.3f}"
                )
                print(f"  Comentário: {item['text']}")
        except Exception as exc:
            print(f"Erro ao processar a pergunta: {exc}")

if __name__ == "__main__":
    main()
