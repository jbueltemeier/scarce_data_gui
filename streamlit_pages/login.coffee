login
    -> set auth info in session state
    -> read records: open_json_data (load record to label by expert)
    -> data init: load_user_data (put data to session state)
    -> ui init:
        ziel: use session state to check page and use session state to check given data

next
    -> increment cursor
    -> save last record (file or ram?)
    -> use session state to check given data

prev
    -> decrement cursor
    -> save last record
    -> use session state to check given data

---------

datenbeschreibung (Data Description):

daten records: (data.json):

    datei: array -> dictionaries -> results
    cursor: array index + results index

session_state (ram daten) records:

    session_state.data -> records

    cursor:
        session_state.page => calculated from count,rcount
        session_state.count // array index
        session_state.rcount // results index

        increment function and decrement function

        dlen entspricht max page

user label daten: (username.json):

    datei: dictionary -> key: page
    session_state.userdata -> key: page
