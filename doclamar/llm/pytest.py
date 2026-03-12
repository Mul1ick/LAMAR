from doclamar.llm.llm_provider import get_llm

llm = get_llm()

print(
    llm.generate(
        "Explain machine learning in 3 bullet points."
    )
)
