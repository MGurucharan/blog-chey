import { useState } from "react";
import { NavLink } from "react-router-dom";

const CreateNavbar = () => {
  const [isOpen, setOpen] = useState(false);

  const toggleNavbar = () => {
    setOpen(!isOpen);
  };

  return (
    <nav className="p-5 shrink-0 w-full fixed z-100 bg-[#0e100f] h-20">
      <div className="flex justify-between items-center h-10 overflow-visible">
        <div className="shrink-0 ml-4 md:ml-10">
          <NavLink to="/" className="focus:outline-none">
            <img
              className="font-semibold object-contain w-35 h-35"
              src="./public/blogchey.png"
              alt="logo"
            />
          </NavLink>
        </div>

        <div className="hidden md:flex items-center gap-7 mr-10">
          <NavLink
            className="text-[#7C7C6F] text-[20px] font-semibold focus:outline-none hover:text-white transition duration-300"
            to="/profile"
          >
            Profile
          </NavLink>
          <NavLink
            className="text-[#7C7C6F] text-[20px] font-semibold focus:outline-none hover:text-white transition duration-300"
            to="/drafts"
          >
            Drafts
          </NavLink>
        </div>

        <button
          onClick={toggleNavbar}
          className="shrink-0 block mr-4 md:hidden focus:outline-none"
          aria-label="Toggle menu"
        >
          <img
            className="cursor-pointer w-6 h-6"
            src="./public/more.png"
            alt="Menu"
          />
        </button>
      </div>

      {isOpen && (
        <div className="md:hidden mt-3 flex flex-col gap-4 ml-4 z-1 bg-[#0e100f] py-3 px-3 rounded-lg">w
          <NavLink
            className="text-[#7C7C6F] text-[18px] hover:text-white transition duration-300"
            to="/profile"
          >
            Profile
          </NavLink>
          <div className="flex gap-4 items-center">
            <NavLink
              className="text-[#7C7C6F] text-[18px] hover:text-white transition duration-300"
              to="/drafts"
            >
              Drafts
            </NavLink>
          </div>
        </div>
      )}

      {/* <hr className="border-0.005 border-[#5b5b55] mt-0 mx-9 md:mx-10" /> */}
    </nav>
  );
};

export default CreateNavbar;
