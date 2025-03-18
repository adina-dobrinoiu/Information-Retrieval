class CISILoader:
    """Class responsible for loading the CISI dataset: documents, queries, and relevance judgments."""

    @staticmethod
    def load_documents(path):
        doc_set = {}
        doc_id, doc_text = "", ""

        with open(path) as f:
            lines = ""
            for l in f.readlines():
                lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
            lines = lines.lstrip("\n").split("\n")

        for l in lines:
            if l.startswith(".I"):
                doc_id = int(l.split(" ")[1].strip())
            elif l.startswith(".X"):
                doc_set[doc_id] = doc_text.lstrip(" ")
                doc_id, doc_text = "", ""
            else:
                doc_text += l.strip()[3:] + " "

        return doc_set

    @staticmethod
    def load_queries(path):
        qry_set = {}
        qry_id = ""

        with open(path) as f:
            lines = ""
            for l in f.readlines():
                lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
            lines = lines.lstrip("\n").split("\n")

        for l in lines:
            if l.startswith(".I"):
                qry_id = int(l.split(" ")[1].strip())
            elif l.startswith(".W"):
                qry_set[qry_id] = l.strip()[3:]
                qry_id = ""

        return qry_set

    @staticmethod
    def load_relevance(path):
        rel_set = {}

        with open(path) as f:
            for l in f.readlines():
                qry_id = int(l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0])
                doc_id = int(l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1])
                if qry_id in rel_set:
                    rel_set[qry_id].append(doc_id)
                else:
                    rel_set[qry_id] = [doc_id]

        return rel_set