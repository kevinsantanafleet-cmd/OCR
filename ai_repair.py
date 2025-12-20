from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, List, Sequence

import openai
from pydantic import BaseModel


class BankTransaction(BaseModel):
    raw_date: str
    description: str
    mutation: Decimal
    balance: Decimal


class AITransactionResult(BaseModel):
    date: str
    description: str
    mutation_amount: float
    balance_after: float


class RepairResponse(BaseModel):
    transactions: List[AITransactionResult]


@dataclass
class Quarantine:
    raw_data: Iterable[Sequence[str]]


def verify_math_chain(transactions: Sequence[BankTransaction]) -> bool:
    if not transactions:
        return False

    previous_balance = transactions[0].balance
    for tx in transactions[1:]:
        expected_balance = previous_balance + tx.mutation
        if tx.balance != expected_balance:
            return False
        previous_balance = tx.balance
    return True


async def repair_with_openai(
    raw_csv_chunk: str,
    client: openai.AsyncOpenAI | None = None,
    model: str = "gpt-4o",
) -> List[BankTransaction]:
    """
    Fallback: Ask AI to make sense of garbage text.
    """
    if client is None:
        client = openai.AsyncOpenAI()

    prompt = f"""
    You are a Forensic Accountant.
    Reconstruct the bank transactions from this broken OCR text.

    RULES:
    1. Extract Date, Description, Mutation, and Balance.
    2. Merge multi-line descriptions into one line.
    3. If a value is (DB) or negative, ensure mutation_amount is negative.
    4. STRICTLY output JSON matching the schema.

    RAW TEXT:
    {raw_csv_chunk}
    """

    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            functions=[
                {
                    "name": "submit_repair",
                    "parameters": RepairResponse.model_json_schema(),
                }
            ],
            function_call={"name": "submit_repair"},
        )

        args = completion.choices[0].message.function_call.arguments
        ai_data = RepairResponse.model_validate_json(args)

        clean_results = []
        for tx in ai_data.transactions:
            clean_results.append(
                BankTransaction(
                    raw_date=tx.date,
                    description=tx.description,
                    mutation=Decimal(str(tx.mutation_amount)),
                    balance=Decimal(str(tx.balance_after)),
                )
            )
        return clean_results
    except Exception as exc:
        print(f"Even AI failed: {exc}")
        return []


async def robust_extraction(
    raw_data: Iterable[Sequence[str]],
    processor,
    min_confidence: float = 0.5,
) -> List[BankTransaction] | Quarantine:
    """
    Try standard logic first, then fall back to AI repair with strict validation.
    """
    transactions, confidence = processor.stitch_rows(raw_data)

    if confidence < min_confidence or not verify_math_chain(transactions):
        print("Standard Parse Failed. Engaging AI Repair...")
        raw_text_blob = "\n".join([",".join(row) for row in raw_data])
        transactions = await repair_with_openai(raw_text_blob)

        if not verify_math_chain(transactions):
            print("AI Repair Failed Math Check. Quarantining.")
            return Quarantine(raw_data)

    return transactions
