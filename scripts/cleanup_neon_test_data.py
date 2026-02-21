import os

import psycopg
from psycopg.rows import dict_row


SESSION_ID = "00000000-0000-0000-0000-000000000001"
USER_ID = "00000000-0000-0000-0000-000000000002"


def main() -> None:
    database_url = os.environ.get("TURBODIFF_DATABASE_URL")
    if not database_url:
        raise RuntimeError("TURBODIFF_DATABASE_URL is not set")

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM airfoils
                WHERE session_id = %s AND created_by_user_id = %s
                """,
                (SESSION_ID, USER_ID),
            )

            cur.execute(
                """
                DELETE FROM cst
                WHERE id NOT IN (SELECT DISTINCT cst_id FROM airfoils)
                """
            )

            cur.execute(
                "DELETE FROM sessions WHERE id = %s",
                (SESSION_ID,),
            )

            cur.execute(
                "DELETE FROM users WHERE id = %s",
                (USER_ID,),
            )

        conn.commit()


if __name__ == "__main__":
    main()
