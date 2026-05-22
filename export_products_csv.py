import argparse
import csv
import json
import os
import subprocess
from typing import Any, Dict, Iterable, List, Optional


def pick_column(columns: Iterable[str], candidates: List[str], required: bool = True) -> Optional[str]:
    lower_to_real = {c.lower(): c for c in columns}
    for name in candidates:
        real = lower_to_real.get(name.lower())
        if real:
            return real
    if required:
        raise RuntimeError(f"Не удалось найти колонку. Ожидалась одна из: {', '.join(candidates)}")
    return None


def parse_specs(raw_value: Any) -> str:
    if raw_value is None:
        return ""

    if isinstance(raw_value, (bytes, bytearray)):
        raw_value = raw_value.decode("utf-8", errors="ignore")

    if not isinstance(raw_value, str):
        return ""

    raw_value = raw_value.strip()
    if raw_value.upper() == "NULL":
        return ""
    if not raw_value:
        return ""

    try:
        data = json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value

    if not isinstance(data, dict):
        return raw_value

    items = []
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        title = str(value.get("title") or "").strip()
        spec_value = str(value.get("value") or "").strip()
        unit = str(value.get("unit") or "").strip()
        order = value.get("order")
        if not title or not spec_value:
            continue
        rendered = f"{title}: {spec_value}"
        if unit:
            rendered = f"{rendered} {unit}"
        items.append((order if isinstance(order, int) else 999999, str(key), rendered))

    if not items:
        return ""

    items.sort(key=lambda x: (x[0], x[1]))
    return "; ".join(item[2] for item in items)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.upper() == "NULL":
        return ""
    return text


def run_mysql_query(sql: str, db_name: str, db_host: str, db_port: int, db_user: str, db_password: str) -> List[Dict[str, str]]:
    env = os.environ.copy()
    env["MYSQL_PWD"] = db_password

    command = [
        "mysql",
        f"--host={db_host}",
        f"--port={db_port}",
        f"--user={db_user}",
        "--default-character-set=utf8mb4",
        "--batch",
        "--raw",
        db_name,
        "-e",
        sql,
    ]

    result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Ошибка выполнения mysql-запроса:\n"
            f"{result.stderr.strip() or result.stdout.strip() or 'unknown error'}"
        )

    output = result.stdout.strip()
    if not output:
        return []

    lines = output.splitlines()
    headers = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < len(headers):
            parts.extend([""] * (len(headers) - len(parts)))
        row = dict(zip(headers, parts))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт каталога товаров в CSV для shop_bot.")
    parser.add_argument("--output", default="products.csv", help="Путь до выходного CSV")
    parser.add_argument("--limit", type=int, default=0, help="Ограничить число товаров (0 = без лимита)")
    args = parser.parse_args()

    db_host = os.getenv("DB_HOST", "127.0.0.1")
    db_port = int(os.getenv("DB_PORT", "3306"))
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "b2b_portal_prod_3")

    try:
        products_columns = [
            row["Field"]
            for row in run_mysql_query(
                "SHOW COLUMNS FROM products",
                db_name=db_name,
                db_host=db_host,
                db_port=db_port,
                db_user=db_user,
                db_password=db_password,
            )
        ]
        categories_columns = [
            row["Field"]
            for row in run_mysql_query(
                "SHOW COLUMNS FROM categories",
                db_name=db_name,
                db_host=db_host,
                db_port=db_port,
                db_user=db_user,
                db_password=db_password,
            )
        ]
        prices_columns = [
            row["Field"]
            for row in run_mysql_query(
                "SHOW COLUMNS FROM product_prices",
                db_name=db_name,
                db_host=db_host,
                db_port=db_port,
                db_user=db_user,
                db_password=db_password,
            )
        ]

        product_id_col = pick_column(products_columns, ["id"])
        product_uid_col = pick_column(products_columns, ["uid"])
        product_name_col = pick_column(products_columns, ["name", "title", "product_name"])
        product_description_col = pick_column(
            products_columns,
            ["description", "full_description", "short_description", "annotation", "text"],
            required=False,
        )
        product_specs_col = pick_column(
            products_columns,
            [
                "characteristics",
                "specifications",
                "attributes",
                "params",
                "properties",
                "tech_specs",
                "technical_data",
                "techs",
            ],
            required=False,
        )
        product_deleted_col = pick_column(products_columns, ["deleted_at"], required=False)
        product_category_col = pick_column(products_columns, ["breez_category_id"])

        category_join_col = pick_column(categories_columns, ["id", "uid", "breez_category_id"])
        category_name_col = pick_column(categories_columns, ["name", "title"], required=False)

        price_uid_col = pick_column(prices_columns, ["uid", "product_uid"])
        price_value_col = pick_column(prices_columns, ["price", "value", "retail_price", "amount"])
        price_deleted_col = pick_column(prices_columns, ["deleted_at"], required=False)

        product_where = [f"p.{product_category_col} IS NOT NULL"]
        if product_deleted_col:
            product_where.append(f"p.{product_deleted_col} IS NULL")

        price_where = [f"pp.{price_value_col} IS NOT NULL", f"pp.{price_value_col} > 0"]
        if price_deleted_col:
            price_where.append(f"pp.{price_deleted_col} IS NULL")

        select_fields = [
            f"p.{product_id_col} AS product_id",
            f"p.{product_uid_col} AS product_uid",
            f"p.{product_name_col} AS product_name",
            "pr.price AS product_price",
        ]
        if product_description_col:
            select_fields.append(f"p.{product_description_col} AS product_description")
        else:
            select_fields.append("NULL AS product_description")

        if product_specs_col:
            select_fields.append(f"p.{product_specs_col} AS product_specs")
        else:
            select_fields.append("NULL AS product_specs")

        if category_name_col:
            select_fields.append(f"c.{category_name_col} AS category_name")
        else:
            select_fields.append("NULL AS category_name")

        limit_clause = f"LIMIT {args.limit}" if args.limit and args.limit > 0 else ""

        sql = f"""
            SELECT
                {", ".join(select_fields)}
            FROM products p
            INNER JOIN (
                SELECT
                    pp.{price_uid_col} AS uid,
                    MAX(pp.{price_value_col}) AS price
                FROM product_prices pp
                WHERE {" AND ".join(price_where)}
                GROUP BY pp.{price_uid_col}
            ) pr ON pr.uid = p.{product_uid_col}
            LEFT JOIN categories c
                ON c.{category_join_col} = p.{product_category_col}
            WHERE {" AND ".join(product_where)}
            {limit_clause}
        """

        rows = run_mysql_query(
            sql,
            db_name=db_name,
            db_host=db_host,
            db_port=db_port,
            db_user=db_user,
            db_password=db_password,
        )

        output_rows: List[Dict[str, Any]] = []
        for row in rows:
            name = as_text(row.get("product_name"))
            if not name:
                continue

            description_parts = []
            base_description = as_text(row.get("product_description"))
            if base_description:
                description_parts.append(base_description)

            category_name = as_text(row.get("category_name"))
            if category_name:
                description_parts.append(f"Категория: {category_name}")

            specs_text = parse_specs(row.get("product_specs"))
            if specs_text:
                description_parts.append(f"Характеристики: {specs_text}")

            description = ". ".join(part for part in description_parts if part).strip()
            if not description:
                description = "Описание отсутствует."

            price_raw = as_text(row.get("product_price"))
            if not price_raw:
                continue
            try:
                price = float(price_raw)
                if price.is_integer():
                    price = int(price)
            except ValueError:
                price = price_raw

            output_rows.append(
                {
                    "id": row.get("product_id"),
                    "name": name,
                    "description": description,
                    "price": price,
                }
            )

        with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name", "description", "price"])
            writer.writeheader()
            writer.writerows(output_rows)

        print(f"Готово. Экспортировано {len(output_rows)} товаров в {args.output}")
    except FileNotFoundError as exc:
        raise RuntimeError("Не найден mysql-клиент. Установите пакет mariadb-client/mysql-client.") from exc


if __name__ == "__main__":
    main()
