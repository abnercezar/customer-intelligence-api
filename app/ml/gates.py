"""Regras para gravar um artefato. A contagem de clientes não aprova o modelo."""

MIN_TRAIN_CUSTOMERS = 200
MIN_TEST_CUSTOMERS = 200
MIN_AUC = 0.65
MIN_ACCURACY_LIFT = 0.05
CHURN_RATE_MIN = 0.05
CHURN_RATE_MAX = 0.95
MIN_SEGMENT_CUSTOMERS = 50

SEGMENT_NAMES = ("at_risk", "champion", "loyal", "new", "potential")


def acceptance_failures(report: dict, segment_sizes) -> list:
    """Devolve os motivos para não substituir o artefato. Lista vazia = pode gravar."""
    failures = []

    train_customers = int(report.get("train_customers") or 0)
    test_customers = int(report.get("test_customers") or 0)
    if train_customers < MIN_TRAIN_CUSTOMERS:
        failures.append(
            f"{train_customers} clientes no treino "
            f"(mínimo {MIN_TRAIN_CUSTOMERS} para o número do teste significar alguma coisa)"
        )
    if test_customers < MIN_TEST_CUSTOMERS:
        failures.append(
            f"{test_customers} clientes no corte de teste (mínimo {MIN_TEST_CUSTOMERS})"
        )

    rate = report.get("test_churn_rate")
    if rate is None or not (CHURN_RATE_MIN <= rate <= CHURN_RATE_MAX):
        shown = "indisponível" if rate is None else f"{rate:.1%}"
        failures.append(
            f"taxa de churn no teste: {shown} "
            f"(precisa estar entre {CHURN_RATE_MIN:.0%} e {CHURN_RATE_MAX:.0%})"
        )

    auc = report.get("auc")
    if auc is None or auc < MIN_AUC:
        shown = "indisponível" if auc is None else f"{auc:.3f}"
        failures.append(f"ROC AUC no teste: {shown} (mínimo {MIN_AUC:.2f})")

    accuracy = report.get("accuracy")
    baseline = report.get("baseline")
    if accuracy is None or baseline is None or accuracy < baseline + MIN_ACCURACY_LIFT:
        if accuracy is None or baseline is None:
            failures.append("acurácia do teste indisponível")
        else:
            failures.append(
                f"acurácia {accuracy:.1%} contra baseline {baseline:.1%} "
                f"(precisa ganhar por pelo menos {MIN_ACCURACY_LIFT:.0%})"
            )

    if not segment_sizes:
        note = report.get("segment_note") or "segmentos não calculados"
        failures.append(note)
        return failures

    for name in SEGMENT_NAMES:
        count = int(segment_sizes.get(name, 0))
        if count < MIN_SEGMENT_CUSTOMERS:
            failures.append(
                f"segmento {name} tem {count} clientes (mínimo {MIN_SEGMENT_CUSTOMERS})"
            )

    return failures
