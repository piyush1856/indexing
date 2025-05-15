import shutil

import subprocess
import os

import hashlib

from indexing.serializers import SourceItemKind
from indexing.utils import is_ignored

from urllib.parse import urlparse
import re
import uuid
import httpx
import asyncio
from abc import ABC, abstractmethod
from base64 import b64encode
from bs4 import BeautifulSoup


class DataExtractor(ABC):
    @abstractmethod
    async def extract(self, **kwargs) -> list[dict]:
        """Fetch data from the source and return a list of records."""
        pass

    @abstractmethod
    async def is_project_public(self) -> bool:
        pass

    @abstractmethod
    async def validate_credentials(self) -> bool:
        pass


class GitRepoExtractor(DataExtractor):
    """Base class for Git-based repository extractors."""

    def __init__(self, repo_url, branch_name=None, pat=None):
        super().__init__()
        self.repo_url = repo_url
        self.branch_name = branch_name
        self.pat = pat
        self.temp_folder_path = os.path.expanduser('~/repo_temp')

        # Validate URL and set clone URL in child classes
        self._validate_url()
        self.clone_url = self._get_clone_url()

    @abstractmethod
    def _validate_url(self):
        """Validate the repository URL format."""
        pass

    @abstractmethod
    def _get_clone_url(self) -> str:
        """Get the clone URL with authentication."""
        pass

    async def clone_repo(self, repo_path=None, repo_name=None) -> str:
        try:
            if not repo_name:
                repo_name = str(uuid.uuid4())
            if not repo_path:
                repo_path = os.path.join(self.temp_folder_path, repo_name)

            process = await asyncio.create_subprocess_exec(
                'git', 'clone', '--depth', '1', '-b', self.branch_name, self.clone_url, repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise Exception(f"Git clone failed with error: {stderr.decode()}")
        except Exception as e:
            raise Exception(f"Error cloning repository: {e}")
        return repo_path

    def get_commit_sha(self, repo_path: str) -> str:
        result = subprocess.run(
            ['git', '-C', repo_path, 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None

    def get_git_blob_sha(self, repo_path: str, file_path: str) -> str:
        """Returns blob SHA for a file in a Git repo"""
        rel_path = os.path.relpath(file_path, repo_path)
        result = subprocess.run(
            ['git', '-C', repo_path, 'ls-tree', 'HEAD', rel_path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.strip().split()[2]  # blob SHA
        return None

    async def extract(self, **kwargs) -> list[dict]:
        """
        Clones the repo and extracts files, returning a list of file content records with metadata.
        """
        cloned_repo_path = None
        file_content_dict = []
        try:
            cloned_repo_path = await self.clone_repo()
            commit_sha = self.get_commit_sha(cloned_repo_path)

            for root, _, repo_files in os.walk(cloned_repo_path):
                for repo_file in repo_files:
                    file_path = os.path.join(root, repo_file)
                    if is_ignored(file_path):
                        continue
                    relative_path = os.path.relpath(file_path, cloned_repo_path)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if content.strip():
                        blob_sha = self.get_git_blob_sha(cloned_repo_path, file_path)
                        file_content_dict.append({
                            "path": relative_path,
                            "content": content,
                            "version_tag": commit_sha,
                            "provider_item_id": blob_sha,
                            "checksum": hashlib.sha256(content.encode()).hexdigest(),
                            "uuid": str(uuid.uuid4()),
                            "kind": SourceItemKind.file.value
                        })
            return file_content_dict
        except Exception as e:
            raise Exception(f"Error extracting repo contents: {e}")
        finally:
            if cloned_repo_path:
                shutil.rmtree(cloned_repo_path)


class GitLabRepoExtractor(GitRepoExtractor):
    def _validate_url(self):
        if not re.match(r"^https://gitlab\.com/[^/]+/[^/]+$", self.repo_url):
            raise ValueError(f"Invalid GitLab repository URL")

    def _get_clone_url(self) -> str:
        return f"https://oauth2:{self.pat}@gitlab.com/{self.repo_url.split('gitlab.com/')[1]}"

    async def is_project_public(self) -> bool:
        """Check if GitLab repository is public."""
        try:
            # Extract owner and repo name from URL
            path_parts = self.repo_url.split('gitlab.com/')[1].split('/')
            project_path = '/'.join(path_parts)

            url = f"https://gitlab.com/api/v4/projects/{project_path.replace('/', '%2F')}"

            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                if response.status_code == 200:
                    project_data = response.json()
                    return project_data.get("visibility", "") == "public"
                elif response.status_code == 404:
                    return False
                else:
                    return False
        except Exception as e:
            return False

    async def validate_credentials(self) -> dict:
        """
        Validate URL, access token, and branch existence.
        Returns a dictionary with validation status and messages.
        """
        result = {
            "is_valid": False,
            "url_valid": False,
            "token_valid": False,
            "branch_valid": False,
            "message": ""
        }

        try:
            # Validate URL format and parse components
            if not self.repo_url:
                result["message"] = "Repository URL is required"
                return result

            try:
                path_parts = self.repo_url.split('gitlab.com/')[1].split('/')
                project_path = '/'.join(path_parts)
                result["url_valid"] = True
            except (IndexError, AttributeError):
                result["message"] = "Invalid repository URL format"
                return result

            # Prepare headers for API calls
            headers = {}
            if self.pat:
                headers["Authorization"] = f"Bearer {self.pat}"

            # Check repository existence and token validity
            async with httpx.AsyncClient() as client:
                repo_url = f"https://gitlab.com/api/v4/projects/{project_path.replace('/', '%2F')}"
                response = await client.get(repo_url, headers=headers)

                if response.status_code == 401:
                    result["message"] = "Invalid access token or insufficient permissions"
                    return result
                elif response.status_code == 404:
                    result["message"] = f"Repository '{project_path}' not found"
                    return result
                elif response.status_code != 200:
                    result["message"] = f"Failed to access repository: {response.status_code}"
                    return result

                result["token_valid"] = True

                # Check branch existence
                if self.branch_name:
                    branch_url = f"https://gitlab.com/api/v4/projects/{project_path.replace('/', '%2F')}/repository/branches/{self.branch_name}"
                    branch_response = await client.get(branch_url, headers=headers)

                    if branch_response.status_code != 200:
                        result["message"] = f"Branch '{self.branch_name}' not found in repository"
                        return result

                    result["branch_valid"] = True
                else:
                    result["message"] = "Branch name is required"
                    return result

                # All validations passed
                result["is_valid"] = True
                result["message"] = "All credentials are valid"
                return result

        except Exception as e:
            result["message"] = f"Error validating credentials: {str(e)}"
            return result


class GitHubRepoExtractor(GitRepoExtractor):
    def _validate_url(self):
        if not re.match(r"^https://github\.com/[^/]+/[^/]+$", self.repo_url):
            raise ValueError(f"Invalid GitHub repository URL")

    def _get_clone_url(self) -> str:
        # For public repos, we can use the regular HTTPS URL if no token is provided
        if not self.pat:
            return self.repo_url
        return f"https://{self.pat}@github.com/{self.repo_url.split('github.com/')[1]}"

    async def is_project_public(self) -> bool:
        """Check if GitHub repository is public."""
        try:
            # Extract owner and repo name from URL
            path_parts = self.repo_url.split('github.com/')[1].split('/')
            owner, repo = path_parts[0], path_parts[1]

            url = f"https://api.github.com/repos/{owner}/{repo}"

            headers = {}
            if self.pat:
                headers["Authorization"] = f"token {self.pat}"

            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    repo_data = response.json()
                    return not repo_data.get("private", True)
                elif response.status_code == 404:
                    return False
                else:
                    return False
        except Exception as e:
            return False

    async def validate_credentials(self) -> dict:
        """
        Validate URL, access token, and branch existence.
        Returns a dictionary with validation status and messages.
        """
        result = {
            "is_valid": False,
            "url_valid": False,
            "token_valid": False,
            "branch_valid": False,
            "message": ""
        }

        try:
            # Validate URL format and parse components
            if not self.repo_url:
                result["message"] = "Repository URL is required"
                return result

            try:
                path_parts = self.repo_url.split('github.com/')[1].split('/')
                owner, repo = path_parts[0], path_parts[1]
                result["url_valid"] = True
            except (IndexError, AttributeError):
                result["message"] = "Invalid repository URL format"
                return result

            # Prepare headers for API calls
            headers = {}
            if self.pat:
                headers["Authorization"] = f"token {self.pat}"

            # Check repository existence and if it's public
            async with httpx.AsyncClient() as client:
                repo_url = f"https://api.github.com/repos/{owner}/{repo}"
                response = await client.get(repo_url, headers=headers)

                # Debug information
                print(f"API Response: {response.status_code}")
                if response.status_code != 200:
                    print(f"Error response: {response.text}")

                if response.status_code == 404:
                    result["message"] = f"Repository '{owner}/{repo}' not found"
                    return result
                elif response.status_code == 401:
                    result["message"] = "Invalid access token"
                    return result
                elif response.status_code != 200:
                    result["message"] = f"Failed to access repository: {response.status_code} - {response.text}"
                    return result

                # Check if repo is public
                repo_data = response.json()
                is_public = not repo_data.get("private", True)

                # For private repos, we need a valid token
                if not is_public:
                    if not self.pat:
                        result["message"] = "Private repository requires an access token"
                        return result
                    result["token_valid"] = True
                else:
                    # Public repo doesn't need token validation
                    result["token_valid"] = True

                # Check branch existence
                if self.branch_name:
                    branch_url = f"https://api.github.com/repos/{owner}/{repo}/branches/{self.branch_name}"
                    branch_response = await client.get(branch_url, headers=headers)

                    if branch_response.status_code != 200:
                        result["message"] = f"Branch '{self.branch_name}' not found in repository"
                        return result

                    result["branch_valid"] = True
                else:
                    result["message"] = "Branch name is required"
                    return result

                # All validations passed
                result["is_valid"] = True
                result["message"] = "All credentials are valid"
                return result

        except Exception as e:
            result["message"] = f"Error validating credentials: {str(e)}"
            return result


class AzureDevopsRepoExtractor(GitRepoExtractor):
    def _validate_url(self):
        if not re.match(r"^https://[^@]+@dev\.azure\.com/.+/.+/_git/.+$", self.repo_url):
            raise ValueError(f"Invalid Azure DevOps repository URL")

    def _get_clone_url(self) -> str:
        try:
            return f"https://{self.pat}@{self.repo_url.split('@')[1]}"
        except IndexError:
            raise ValueError("Malformed Azure DevOps URL, missing '@' in the expected format")

    async def is_project_public(self) -> bool:
        """Check if Azure DevOps project is public."""
        try:
            parsed = urlparse("https://" + self.repo_url.split("@")[1])
            path_parts = [p for p in parsed.path.strip("/").split("/") if p]
            if len(path_parts) < 2:
                return False

            organization, project = path_parts[0], path_parts[1]
            url = f"https://dev.azure.com/{organization}/_apis/projects/{project}?api-version=6.0"

            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                if response.status_code == 200:
                    project_data = response.json()
                    visibility = project_data.get("visibility", "").lower()
                    return visibility == "public"
                elif response.status_code == 404:
                    return False
                else:
                    return False
        except Exception as e:
            return False

    async def validate_credentials(self) -> dict:
        """
        Validate URL, PAT, and branch existence.
        Returns a dictionary with validation status and messages.
        """
        result = {
            "is_valid": False,
            "url_valid": False,
            "token_valid": False,
            "branch_valid": False,
            "message": ""
        }

        try:
            # Validate URL format and parse components
            if not self.repo_url:
                result["message"] = "Repository URL is required"
                return result

            try:
                parsed = urlparse("https://" + self.repo_url.split("@")[1])
                path_parts = [p for p in parsed.path.strip("/").split("/") if p]
                organization, project = path_parts[0], path_parts[1]
                repo_name = path_parts[-1]
                result["url_valid"] = True
            except (IndexError, AttributeError):
                result["message"] = "Invalid repository URL format"
                return result

            # Prepare headers for API calls
            headers = {}
            if self.pat:  # Use pat instead of pat
                auth_str = b64encode(f":{self.pat}".encode()).decode()
                headers["Authorization"] = f"Basic {auth_str}"

            async with httpx.AsyncClient() as client:
                # Check repository existence and PAT validity
                repo_url = f"https://dev.azure.com/{organization}/{project}/_apis/git/repositories/{repo_name}?api-version=6.0"
                response = await client.get(repo_url, headers=headers)

                if response.status_code == 302:
                    result["message"] = "Invalid access token"
                    return result
                elif response.status_code == 404:
                    result["message"] = f"Repository '{repo_name}' not found in {organization}/{project}"
                    return result
                elif response.status_code != 200:
                    result["message"] = f"Failed to access repository."
                    return result

                result["token_valid"] = True

                # Check branch existence
                if self.branch_name:
                    branch_url = f"https://dev.azure.com/{organization}/{project}/_apis/git/repositories/{repo_name}/refs?filter=heads/{self.branch_name}&api-version=6.0"
                    branch_response = await client.get(branch_url, headers=headers)

                    if branch_response.status_code != 200:
                        result["message"] = f"Failed to verify branch. Status code: {branch_response.status_code}"
                        return result

                    branches = branch_response.json().get("value", [])
                    if not branches:
                        result["message"] = f"Branch '{self.branch_name}' not found in repository"
                        return result

                    result["branch_valid"] = True
                else:
                    result["message"] = "Branch name is required"
                    return result

                # All validations passed
                result["is_valid"] = True
                result["message"] = "All credentials are valid"
                return result

        except Exception as e:
            result["message"] = f"Error validating credentials: {str(e)}"
            return result


class QuipExtractor(DataExtractor):
    def __init__(self, pat: str, urls: list[str], max_docs_per_kb: int = 10):
        self.pat = pat
        self.urls = urls
        self.max_docs_per_kb = max_docs_per_kb
        self.base_api_url = "https://platform.quip.com/1"
        self.headers = {"Authorization": f"Bearer {self.pat}"}
        self.semaphore = asyncio.Semaphore(5)  # Reduced for stability
        self.folder_name_cache = {}  # Cache folder names
        self.thread_title_cache = {}  # Cache thread titles
        self.folder_content_cache = {}  # Cache folder contents
        self.client = httpx.AsyncClient(timeout=30.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def extract(self, **kwargs) -> list[dict]:
        """Fetch data from Quip and return a list of records."""
        try:
            max_total_docs = self.max_docs_per_kb  # Overall maximum docs

            thread_ids = []
            folder_mapping = {}  # Map thread_ids to their folder paths and titles

            # Process each URL sequentially instead of in parallel
            for url in self.urls:
                await self._process_url(url, folder_mapping, thread_ids, max_total_docs)

            # Process all collected thread IDs sequentially
            all_records = []

            for thread_id in thread_ids:
                # Use the folder mapping if available
                mapping = folder_mapping.get(thread_id, {"path": "", "title": ""})
                folder_path = mapping["path"]
                thread_title = mapping["title"] or self.thread_title_cache.get(thread_id, "")
                source_url = mapping.get("source_url", "")  # Get the source URL from mapping

                record = await self._get_thread_content(thread_id, folder_path, thread_title, source_url)
                if record:
                    all_records.append(record)

            return all_records
        finally:
            await self.client.aclose()

    def _extract_id_from_url(self, url: str) -> str | None:
        """Extract Quip ID from URL, handling both document and folder URLs"""
        try:
            parsed = urlparse(url)
            path = parsed.path.strip("/")

            # We need just the ID part
            parts = path.split("/")

            # The ID is always the first part after the domain
            quip_id = parts[0] if parts else None

            return quip_id
        except Exception as e:
            return None

    async def _get_item_type(self, quip_id: str) -> str | None:
        """Determine if it's a folder or thread"""
        async with self.semaphore:
            folder_url = f"{self.base_api_url}/folders/{quip_id}"
            thread_url = f"{self.base_api_url}/threads/{quip_id}"

            # Send both requests concurrently
            try:
                folder_task = self.client.get(folder_url, headers=self.headers)
                thread_task = self.client.get(thread_url, headers=self.headers)

                folder_resp, thread_resp = await asyncio.gather(folder_task, thread_task)

                if folder_resp.status_code == 200:
                    return "folder"
                if thread_resp.status_code == 200:
                    return "thread"



                # Check if this might be a thread ID with additional path components
                # Try with just the first part of the ID
                if "/" in quip_id:
                    base_id = quip_id.split("/")[0]
                    return await self._get_item_type(base_id)

                return None
            except Exception as e:
                return None

    async def _get_folder_name(self, folder_id: str) -> str:
        """Get the name of a folder with caching"""
        # Check cache first
        if folder_id in self.folder_name_cache:
            return self.folder_name_cache[folder_id]

        try:
            async with self.semaphore:
                folder_url = f"{self.base_api_url}/folders/{folder_id}"
                response = await self.client.get(folder_url, headers=self.headers)
                if response.status_code == 200:
                    folder_data = response.json()
                    # Cache the folder content for later use
                    self.folder_content_cache[folder_id] = folder_data

                    # Try to get title from different places in the response
                    folder_name = folder_data.get("title")

                    # If no title directly, try to get from folder or thread objects
                    if not folder_name and "folder" in folder_data:
                        folder_name = folder_data["folder"].get("title")

                    if not folder_name and "thread" in folder_data:
                        folder_name = folder_data["thread"].get("title")

                    # If we found a name, cache it and return
                    if folder_name:
                        # Clean the folder name for filesystem safety
                        safe_folder_name = re.sub(r'[\\/*?:"<>|]', "-", folder_name)
                        self.folder_name_cache[folder_id] = safe_folder_name
                        return safe_folder_name

                    # If we get here, we couldn't find a title
                    return f"Folder-{folder_id[:8]}"
                else:
                    return f"Folder-{folder_id[:8]}"
        except Exception as e:
            return f"Folder-{folder_id[:8]}"

    async def _process_url(self, url, folder_mapping, thread_ids, max_total_docs):
        """Process a single URL (document or folder)"""
        quip_id = self._extract_id_from_url(url)
        if not quip_id:
            return


        item_type = await self._get_item_type(quip_id)

        # If it's a document, add it directly
        if item_type == "thread":
            thread_title = await self._get_thread_title(quip_id)
            folder_mapping[quip_id] = {
                "path": "",  # Root path
                "title": thread_title or f"Document-{quip_id[:8]}",
                "source_url": url  # Store the original URL
            }
            thread_ids.append(quip_id)

        # If it's a folder, process it
        elif item_type == "folder":
            folder_name = await self._get_folder_name(quip_id)
            if not folder_name:
                # Try to extract folder name from URL if API doesn't provide it
                url_parts = url.split("/")
                if len(url_parts) > 4:  # Has a path component after the ID
                    folder_name = url_parts[-1]
                else:
                    folder_name = f"Folder-{quip_id[:8]}"


            # Process the folder recursively with global document limit
            await self._process_folder_recursively(
                quip_id,
                folder_name,
                folder_mapping,
                thread_ids,
                max_total_docs,
                url  # Pass the original URL
            )
        else:
            pass

    async def _process_folder_recursively(self, folder_id, folder_name, folder_mapping, thread_ids, max_total_docs,
                                          source_url):
        """Process a folder and its subfolders recursively"""
        # Stop if we've already reached the maximum total documents
        if len(thread_ids) >= max_total_docs:
            return

        # Check if we already have the folder content cached
        folder_data = self.folder_content_cache.get(folder_id)

        if not folder_data:
            try:
                async with self.semaphore:
                    folder_url = f"{self.base_api_url}/folders/{folder_id}"
                    response = await self.client.get(folder_url, headers=self.headers)
                    if response.status_code != 200:
                        return
                    folder_data = response.json()
                    # Cache the folder data for future use
                    self.folder_content_cache[folder_id] = folder_data
            except Exception as e:
                return

        # Process threads and subfolders
        children = folder_data.get("children", [])

        thread_children = [child for child in children if child.get("thread_id")]
        folder_children = [child for child in children if child.get("folder_id")]


        # Process threads first (they're usually faster)
        for child in thread_children:
            if len(thread_ids) >= max_total_docs:
                break

            thread_id = child["thread_id"]
            thread_title = None

            # Try to get title from the child data first
            if "thread" in child and child["thread"].get("title"):
                thread_title = child["thread"]["title"]
                self.thread_title_cache[thread_id] = thread_title

            # If no title, make a separate API call to get it
            if not thread_title:
                thread_title = await self._get_thread_title(thread_id)

            # Use a more descriptive fallback if still no title
            if not thread_title:
                thread_title = f"Document-{thread_id[:8]}"

            # Store thread info for later processing
            folder_mapping[thread_id] = {
                "path": folder_name,
                "title": thread_title,
                "source_url": source_url  # Store the original URL
            }
            thread_ids.append(thread_id)



        # Process subfolders if we haven't reached the limit
        if len(thread_ids) < max_total_docs:
            for child in folder_children:
                child_folder_id = child["folder_id"]

                # Try to get folder name from the child data first
                child_folder_name = None
                if "folder" in child and child["folder"].get("title"):
                    child_folder_name = child["folder"]["title"]
                elif "thread" in child and child["thread"].get("title"):
                    child_folder_name = child["thread"]["title"]

                # If no name in child data, check our cache
                if not child_folder_name and child_folder_id in self.folder_name_cache:
                    child_folder_name = self.folder_name_cache[child_folder_id]

                # If still no name, make a dedicated API call
                if not child_folder_name:
                    child_folder_name = await self._get_folder_name(child_folder_id)

                # Use a unique fallback if we still don't have a name
                if not child_folder_name:
                    child_folder_name = f"Folder-{child_folder_id[:8]}"

                # Clean the folder name for filesystem safety
                child_folder_name = re.sub(r'[\\/*?:"<>|]', "-", child_folder_name)

                # Cache the folder name
                self.folder_name_cache[child_folder_id] = child_folder_name

                # Create the nested path
                nested_path = f"{folder_name}/{child_folder_name}" if folder_name else child_folder_name

                # Process this subfolder
                await self._process_folder_recursively(
                    child_folder_id,
                    nested_path,
                    folder_mapping,
                    thread_ids,
                    max_total_docs,
                    source_url  # Pass the original URL
                )

                # Stop if we've reached the limit
                if len(thread_ids) >= max_total_docs:
                    break

    async def _get_thread_title(self, thread_id: str) -> str:
        """Get just the title of a thread with caching"""
        # Check cache first
        if thread_id in self.thread_title_cache:
            return self.thread_title_cache[thread_id]

        try:
            async with self.semaphore:
                response = await self.client.get(f"{self.base_api_url}/threads/{thread_id}", headers=self.headers)
                if response.status_code == 200:
                    thread = response.json()

                    # Try to get title from thread metadata
                    title = thread.get("thread", {}).get("title", "")

                    # If no title in metadata, try to extract from HTML content
                    if not title and "html" in thread:
                        try:
                            soup = BeautifulSoup(thread["html"], 'html.parser')

                            # Try to find title in this order: h1, h2, h3, first paragraph
                            for tag in ['h1', 'h2', 'h3']:
                                element = soup.find(tag)
                                if element and element.text.strip():
                                    title = element.text.strip()
                                    break

                            # If no headings, try first paragraph
                            if not title:
                                p = soup.find('p')
                                if p and p.text.strip():
                                    text = p.text.strip()
                                    # Limit paragraph text to reasonable length
                                    if len(text) > 40:
                                        title = text[:40] + "..."
                                    else:
                                        title = text
                        except Exception as e:
                            pass

                    # Store in cache if we found a title
                    if title:
                        self.thread_title_cache[thread_id] = title

                    return title
                return ""
        except Exception as e:
            return ""

    async def _get_thread_content(self, thread_id, folder_path="", thread_title="", source_url=""):
        """Get the content of a thread with rate limiting"""
        try:
            # Check if we already have the title in cache
            if not thread_title and thread_id in self.thread_title_cache:
                thread_title = self.thread_title_cache[thread_id]

            async with self.semaphore:
                response = await self.client.get(f"{self.base_api_url}/threads/{thread_id}", headers=self.headers)
                if response.status_code != 200:
                    return None

                thread = response.json()
                content = thread.get("html", "")
                if not content.strip():
                    return None

                # Use provided thread title or get from response
                doc_title = thread_title
                if not doc_title:
                    # Try to get title from thread metadata
                    doc_title = thread.get("thread", {}).get("title", "")

                    # If no title in metadata, try to extract from HTML content
                    if not doc_title and content:
                        try:
                            soup = BeautifulSoup(content, 'html.parser')
                            for tag in ['h1', 'h2', 'h3']:
                                element = soup.find(tag)
                                if element and element.text.strip():
                                    doc_title = element.text.strip()
                                    break
                        except Exception as e:
                            pass

                # Use fallback if still no title - include thread ID for uniqueness
                if not doc_title:
                    doc_title = f"Document-{thread_id[:8]}"

                # Clean up filename to be safe for filesystems
                safe_filename = re.sub(r'[\\/*?:"<>|]', "-", doc_title)
                # Remove HTML tags if any remain in the title
                safe_filename = re.sub(r'<[^>]*>', '', safe_filename)
                # Ensure filename is not too long
                if len(safe_filename) > 100:
                    safe_filename = safe_filename[:97] + "..."

                path = f"{folder_path}/{safe_filename}.html" if folder_path else f"{safe_filename}.html"
                path = path.strip("/").replace("//", "/")

                return {
                    "path": path,
                    "content": content,
                    "version_tag": thread.get("thread", {}).get("updated_usec"),
                    "provider_item_id": thread_id,
                    "checksum": hashlib.sha256(content.encode()).hexdigest(),
                    "uuid": str(uuid.uuid4()),
                    "kind": SourceItemKind.file.value,
                    "source_url": source_url  # Include the original URL
                }
        except Exception as e:
            return None

    async def is_project_public(self) -> bool:
        """Check if Quip content is public."""
        # Quip content is never public without authentication
        return False

    async def validate_credentials(self) -> dict:
        """Validate the Quip access token."""
        result = {
            "is_valid": False,
            "message": "Failed to validate Quip credentials"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_api_url}/users/current", headers=self.headers)
                if response.status_code == 200:
                    result["is_valid"] = True
                    result["message"] = "Token is valid"
                else:
                    result["message"] = f"Auth failed: {response.text}"
                return result
        except Exception as e:
            result["message"] = str(e)
            return result