from plugins.base import IngestPlugin, PluginRegistry
from typing import Dict, Any, List
from pathlib import Path
import os, tempfile, fitz
from docling.document_converter import DocumentConverter
from docling_core.types.doc import PictureItem



@PluginRegistry.register
class DoclingMultimodalIngestPlugin(IngestPlugin):

    # Plugin metadata
    name = "docling_multimodal"
    kind = "file-ingest"
    description = "Ingests text, images and tables content using Docling and CLIP embeddings."
    supported_file_types = {"pdf"}

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "chunk_size": {
                "type": "integer",
                "default": 512,
                "description": "Mida de cada fragment de text si s'utilitza fragmentació per mida."
            },
            "chunk_strategy": {
                "type": "enum",
                "default": "semantic",
                "description": "Estrategia de fragmentació a utilitzar.",
                "options": ["semantic", "fixed"]
            },
            "overlap": {
                "type": "integer",
                "default": 50,
                "description": "Nombre de tokens o unitats que es solapen entre fragments."
            },
            "ocr": {
                "type": "boolean",
                "default": True,
                "description": "Habilitar OCR per a imatges i PDFs que contenen text."
            },
            "include_images": {
                "type": "boolean",
                "default": True,
                "description": "Incloure imatges en els fragments generats."
            },
            "max_pages": {
                "type": "integer",
                "default": 100,
                "description": "Nombre màxim de pàgines a processar en un PDF."
            },
            "language":  {
                "type": "string",
                "default": "en",
                "description": "Idioma del contingut a processar. Utilitzat per a OCR i models de llenguatge."
            }
        }

    def extract_picture_ocr(self, picture: PictureItem, document) -> str:
        """Extrae el OCR (si existe) de los children de un PictureItem"""
        if not picture.children:
            return ""
        ocr_parts = []
        for child in picture.children:
            cref = child.cref
            index = int(cref.split("/")[-1]) if cref.startswith("#/texts/") else None
            if index is not None and 0 <= index < len(document.texts):
                ocr_parts.append(document.texts[index].text)
        return "\n".join(ocr_parts).strip()

    def ingest(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        chunk_size = kwargs.get("chunk_size", 512)
        chunk_strategy = kwargs.get("chunk_strategy", "semantic")
        overlap = kwargs.get("overlap", 50)
        ocr = kwargs.get("ocr", True)
        include_images = kwargs.get("include_images", True)
        max_pages = kwargs.get("max_pages", 100)
        language = kwargs.get("language", "en")
        file_url = kwargs.get("file_url", "")
        VALID_STRATEGIES = ["semantic", "fixed"]

        if chunk_strategy not in VALID_STRATEGIES:
            raise ValueError(f"Invalid chunk_strategy: {chunk_strategy}. Supported strategies: {VALID_STRATEGIES}")

        file_path_obj = Path(file_path)
        file_name = file_path_obj.name

        chunks = []
        common_metadata = {
            "source": file_path,
            "modality": "multimodal",
            "embedding_hint": "clip",
            "converted_with": "Docling",
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "language": language,
            "ocr_enabled": ocr,
            "file_name": file_name,
            "file_url": file_url,
            "original_filename": os.path.basename(file_path),
            "include_images": include_images,
            "overlap": overlap,
            "max_pages": max_pages
        }
        converter = DocumentConverter()

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file_bytes)
            temp_pdf.flush()

            result = converter.convert(
                source=temp_pdf.name,
                max_num_pages=max_pages
            )
            markdown = result.document.export_to_markdown()

        paragraphs = markdown.split("\n\n")
        chunk_idx = 0

        image_refs = []

        if chunk_strategy == "fixed":
            step = chunk_size - overlap
            if step <= 0:
                raise ValueError("Chunk size must be greater than overlap size.")
            for i in range(0, len(paragraphs), step):
                chunk_text = "\n\n".join(paragraphs[i:i + chunk_size])
                if not chunk_text.strip():
                    continue

                has_image = "<!-- image -->" in chunk_text
                if has_image:
                    num_refs = chunk_text.count("<!-- image -->")
                    image_refs.extend([chunk_idx] * num_refs)

                chunk = {
                    "text": chunk_text,
                    "metadata": {
                        **common_metadata,
                        "type": "text",
                        "chunk_index": chunk_idx,
                        "has_image": has_image,
                        "linked_images": ""
                    }
                }
                chunks.append(chunk)
                chunk_idx += 1

        elif chunk_strategy == "semantic":
            chunk = []
            current_length = 0
            for p in paragraphs:
                if not p.strip():
                    continue
                chunk.append(p)
                current_length += len(p.split())
                if current_length >= chunk_size:
                    chunk_text = "\n\n".join(chunk)
                    has_image = "<!-- image -->" in chunk_text
                    if has_image:
                        num_refs = chunk_text.count("<!-- image -->")
                        image_refs.extend([chunk_idx] * num_refs)

                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **common_metadata,
                            "type": "text",
                            "chunk_index": chunk_idx,
                            "has_image": has_image,
                            "linked_images": ""
                        }
                    })
                    chunk_idx += 1
                    chunk = []
                    current_length = 0
            if chunk:
                chunk_text = "\n\n".join(chunk)
                has_image = "<!-- image -->" in chunk_text
                if has_image:
                    num_refs = chunk_text.count("<!-- image -->")
                    image_refs.extend([chunk_idx] * num_refs)

                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **common_metadata,
                        "type": "text",
                        "chunk_index": chunk_idx,
                        "has_image": has_image,
                        "linked_images": ""
                    }
                })

        # Procesamiento de imágenes
        if include_images:
            doc = fitz.open(file_path)
            pdf_path = Path(file_path)
            data_dir = pdf_path.parent
            pdf_stem = Path(kwargs.get("original_filename", pdf_path.stem)).stem
            img_output_dir = Path(data_dir) / "images" / f"imatges_{pdf_stem}"
            img_output_dir.mkdir(parents=True, exist_ok=True)

            ref_index = 0
            picture_counter = 0
            for page_index in range(min(len(doc), max_pages)):
                page = doc[page_index]
                img_info = page.get_images(full=True)
                for img_index, img in enumerate(img_info):
                    xref = img[0]
                    base_img = doc.extract_image(xref)
                    img_bytes = base_img["image"]
                    ext = base_img["ext"]
                    filename = f"image_{page_index}_{img_index}.{ext}"
                    image_path = img_output_dir / filename
                    if ocr and picture_counter < len(result.document.pictures):
                        ocr_text = self.extract_picture_ocr(result.document.pictures[picture_counter], result.document) if ocr else ""
                    else:
                        ocr_text = ""

                    picture_counter += 1

                    try:
                        with open(image_path, "wb") as f:
                            f.write(img_bytes)
                    except Exception as e:
                        print(f"ERROR al guardar la imagen {filename}: {e}")

                    try:
                        relative_url = image_path.relative_to(Path("static").resolve())
                        image_url = f"/static/{relative_url}"
                    except ValueError:
                        image_url = f"/static/{image_path}"

                    linked_chunk_index = image_refs[ref_index] if ref_index < len(image_refs) else None
                    if linked_chunk_index is not None:
                        for chunk in chunks:
                            if chunk["metadata"]["chunk_index"] == linked_chunk_index:
                                chunk["metadata"]["linked_images"] = image_url
                                break

                    image_chunk = {
                        "text": f"Imatge extreta de la pàgina {page_index + 1}, imatge {img_index + 1}.",
                        "metadata": {
                            **common_metadata,
                            "type": "imatge",
                            "chunk_index": chunk_idx,
                            "image_path": image_url,
                            "page_index": page_index,
                            "image_index": img_index,
                            "linked_text_chunk": str(linked_chunk_index) if linked_chunk_index is not None else "",
                            "ocr_text": ocr_text
                        }
                    }
                    chunks.append(image_chunk)
                    chunk_idx += 1
                    ref_index += 1

        for image_chunk in chunks:
            if image_chunk["metadata"].get("type") == "imatge":
                linked_idx = image_chunk["metadata"].get("linked_text_chunk")
                if linked_idx is not None and linked_idx != "":
                    for text_chunk in chunks:
                        if (
                            text_chunk["metadata"].get("type") == "text"
                            and text_chunk["metadata"].get("chunk_index") == int(linked_idx)
                        ):
                            text_chunk["metadata"]["related_image_chunk"] = image_chunk["metadata"]["chunk_index"]
                            text_chunk["metadata"]["related_image_ocr"] = image_chunk["metadata"].get("ocr_text", "")
                            break

        return chunks
